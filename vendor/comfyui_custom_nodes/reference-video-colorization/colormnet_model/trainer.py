"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from colormnet_model.network import ColorMNet

from colormnet_model.losses import LossComputer
from colormnet_util.log_integrator import Integrator
from colormnet_util.image_saver import pool_pairs_221128_TransColorization

# val
from torch.utils.data import DataLoader
from colormnet_inference.data.mask_mapper import MaskMapper
from colormnet_inference.inference_core import InferenceCore
import torch.nn.functional as F
from PIL import Image
from colormnet_util.transforms import lab2rgb_transform_PIL, calculate_psnr


class ColorMNetTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1, wandb = None):
        self.wandb= wandb
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        self.model = nn.parallel.DistributedDataParallel(
            ColorMNet(config).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.model.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.best_psnr = 0
        self.best_it = 0

    def do_val(self, it=0, val_dataset=None):
        self.model.module.eval()
        self.val()

        print('starting validation at it: %s'%(it))
        val_loader = val_dataset.get_datasets()

        config = {}
        config['enable_long_term'] = True
        config['max_mid_term_frames'] = 10
        config['min_mid_term_frames'] = 5
        config['max_long_term_elements'] = 10000
        config['num_prototypes'] = 128
        config['benchmark'] = False
        config['flip'] = False
        config['top_k'] = 30
        config['mem_every'] = 5
        config['deep_update_every'] = -1
        config['hidden_dim'] = 64

        avg_psnr = []
        wb_frames = []

        # with torch.no_grad():
        for vid_reader in val_loader:
            clip_psnr = []
            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
            vid_name = vid_reader.vid_name
            vid_length = len(loader)
            # no need to count usage for LT if the video is not that long anyway
            config['enable_long_term_count_usage'] = (
                config['enable_long_term'] and
                (vid_length
                    / (config['max_mid_term_frames']-config['min_mid_term_frames'])
                    * config['num_prototypes'])
                >= config['max_long_term_elements']
            )

            mapper = MaskMapper()
            processor = InferenceCore(self.model.module, config=config)
            first_mask_loaded = False

            t0 = 0
            for ti, data in enumerate(loader):
                t0 += 1 
                if (t0-1) % 10 != 0: # Do not validate every frame for speed up training
                    continue

                with torch.cuda.amp.autocast(enabled=not config['benchmark']):
                    rgb = data['rgb'].cuda()[0]
                    msk = data.get('mask')
                    msk = msk[:,1:3,:,:] if msk is not None else None

                    info = data['info']
                    frame = info['frame'][0]
                    shape = info['shape']
                    need_resize = info['need_resize'][0]

                    """
                    For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                    Seems to be very similar in testing as my previous timing method 
                    with two cuda sync + time.time() in STCN though 
                    """

                    if not first_mask_loaded:
                        if msk is not None:
                            first_mask_loaded = True
                        else:
                            # no point to do anything without a mask
                            continue

                    if config['flip']:
                        rgb = torch.flip(rgb, dims=[-1])
                        msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                    # Map possibly non-continuous labels to continuous ones
                    if msk is not None:
                        msk = torch.Tensor(msk[0]).cuda()
                        if need_resize:
                            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]

                        processor.set_all_labels(list(range(1,3)))
                        labels = range(1,3)
                    else:
                        labels = None
            
                    # Run the model on this frame
                    # print('******************* START %s *************'%ti)
                    prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))
                    # print('******************* END %s *************'%ti)

                    # Upsample to original size if needed
                    if need_resize:
                        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]
                        rgb = F.interpolate(rgb.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

                    if config['flip']:
                        prob = torch.flip(prob, dims=[-1])

                    # Save the mask
                    if info['save'][0]:
                        out_mask_final = lab2rgb_transform_PIL(torch.cat([rgb[:1,:,:], prob], dim=0))
                        out_mask_final = out_mask_final * 255
                        out_mask_final = out_mask_final.astype(np.uint8)
                        
                        out_img = np.array(Image.fromarray(out_mask_final))
 
                        gt_folder = self.config['validation_root']
                        gt_path = os.path.join(gt_folder, info['vid_name'][0], info['frame'][0])
                        gt_img = np.array(Image.open(gt_path))
                        psnr = calculate_psnr(gt_img, out_img)
                        # avg_psnr.append(psnr)
                        clip_psnr.append(psnr)

                        save_shape = (384, 384) # resize to save wandb space
                        out_img = np.array(Image.fromarray(out_img).resize(save_shape, resample=0))
                        gt_img = np.array(Image.fromarray(gt_img).resize(save_shape, resample=0))

                        wb_frames.append(self.wandb.Image(out_img, caption="Pred_%s_it%s"%(t0, it)))
                        wb_frames.append(self.wandb.Image(gt_img, caption="GT%s_it%s"%(t0, it)))
            self.wandb.log({"val/pairs": wb_frames},step=it)


            print('current item: %s clip_psnr is: %s'%(info['vid_name'][0], np.mean(clip_psnr)))
            avg_psnr += clip_psnr 
                    
        self.wandb.log({'val/psnr': np.mean(avg_psnr)}, step=it)
        self.logger.log_scalar('val/psnr', np.mean(avg_psnr), it)
        print('finish validation at it: %s avg_psnr: %s best_psnr: %s best_it: %s'%(it, np.mean(avg_psnr), self.best_psnr, self.best_it))

        return np.mean(avg_psnr)

    def do_pass(self, data, it=0, val_dataset=None):
        self.model.module.train()
        self.train()

        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        wb_frames = []
        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            key, shrinkage, selection, f16, f8, f4 = self.model('encode_key', frames)

            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))

            v16, hidden = self.model('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            
            values = v16.unsqueeze(3) # add the time dimension

            for ti in range(1, self.num_frames):
                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                # Segment frame ti
                memory_readout = self.model('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                
                # short term memory
                memory_readout_short = self.model('read_memory_short', key[:,:,ti], key[:,:,ti-1], values[:, :, :, ti-1])
                memory_readout += memory_readout_short

                hidden, logits, masks = self.model('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
                        hidden, selector, h_out=(ti < (self.num_frames-1)))

                # No need to encode the last frame
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.model('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits

                if self._is_train:
                    if it % self.log_image_interval == 0 and it != 0:
                        if self.logger is not None:

                            out_mask_final = lab2rgb_transform_PIL(torch.cat([frames[0,ti,:1,:,:], masks[0]], dim=0))
                            out_mask_final = out_mask_final * 255
                            out_mask_final = out_mask_final.astype(np.uint8)
                            out_img = np.array(Image.fromarray(out_mask_final))
                            
                            gt_image_final = lab2rgb_transform_PIL(torch.cat([frames[0,ti,:1,:,:], data['cls_gt'][0,ti]], dim=0))
                            gt_image_final = gt_image_final * 255
                            gt_image_final = gt_image_final.astype(np.uint8)
                            gt_img = np.array(Image.fromarray(gt_image_final))

                            wb_frames.append(self.wandb.Image(out_img, caption="Pred_%s"%ti))
                            wb_frames.append(self.wandb.Image(gt_img, caption="GT_%s"%ti))


            if self._do_log or self._is_train:
                losses = self.loss_computer.compute_l1loss({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)

                    self.wandb.log({'train/loss': losses['total_loss'].item()}, step=it)
                    self.wandb.log({'train/dice_loss_7': losses['dice_loss_7'].item()}, step=it)
                    self.wandb.log({'train/lr': self.scheduler.get_last_lr()[0]}, step=it)

                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384) # resize to save wandb space
                                self.logger.log_cv2('train/pairs', pool_pairs_221128_TransColorization(images, size, num_filled_objects), it)

                                self.wandb.log({"train/pairs": wb_frames},step=it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0 and it >= 129999:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()

        self.scheduler.step()

        # validation
        if it % self.save_network_interval == 0: # log_text_interval
            current_psnr = self.do_val(it, val_dataset=val_dataset)

            if current_psnr >= self.best_psnr:
                self.best_psnr = current_psnr
                self.best_it = it
                self.save_best_network(it)


    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.model.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_best_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_best.pth'
        torch.save(self.model.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.model.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.model.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.model.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.model.eval()
        return self
