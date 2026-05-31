from colormnet_inference.memory_manager import MemoryManager
from colormnet_model.network import ColorMNet
from colormnet_model.aggregate import aggregate

from colormnet_util.tensor_util import pad_divide_by, unpad
import torch

class InferenceCore:
    def __init__(self, network:ColorMNet, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

        self.last_ti_key = None
        self.last_ti_value = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        divide_by = 112 # 16
        image, self.pad = pad_divide_by(image, divide_by)
        image = image.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)

        # segment the current frame is needed
        if need_segment:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)

            # short term memory 
            batch, num_objects, value_dim, h, w = self.last_ti_value.shape
            last_ti_value = self.last_ti_value.flatten(start_dim=1, end_dim=2)
            memory_value_short, _ = self.network.short_term_attn(key, self.last_ti_key, last_ti_value, None, key.shape[-2:])
            memory_value_short = memory_value_short.permute(1, 2, 0).view(batch, num_objects, value_dim, h, w)
            memory_readout += memory_value_short

            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, divide_by)

            pred_prob_with_bg = mask

            self.memory.create_hidden_state(2, key)

        # save as memory if needed
        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg.unsqueeze(0), is_deep_update=is_deep_update)

            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            self.last_ti_key = key
            self.last_ti_value = value

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)

    def step_AnyExemplar(self, image, msk_lll=None, msk_ab=None, valid_labels=None, end=False, flag_FirstframeIsExemplar=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        divide_by = 112 # 16
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, divide_by)
        image = image.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (msk_ab is not None)) and (not end)
        need_segment = (self.curr_ti >= 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels))) if not flag_FirstframeIsExemplar else (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)

        # save as memory if needed
        if msk_ab is not None and not flag_FirstframeIsExemplar:
            need_segment = True
            is_deep_update = False

            msk_lll, _ = pad_divide_by(msk_lll, divide_by)
            msk_lll = msk_lll.unsqueeze(0) # add the batch dimension
            key_mask, shrinkage_mask, selection_mask, f16_mask, f8_mask, f4_mask = self.network.encode_key(msk_lll, 
                                            need_ek=(self.enable_long_term or need_segment), 
                                            need_sk=is_mem_frame)

            msk_ab, _ = pad_divide_by(msk_ab, divide_by)
            pred_prob_with_bg = msk_ab


            self.memory.create_hidden_state(2, key)
        

            value_mask, hidden_mask = self.network.encode_value(msk_lll, f16_mask, self.memory.get_hidden(), 
                                    pred_prob_with_bg.unsqueeze(0), is_deep_update=False)

            # save key-value to memory
            self.memory.add_memory(key_mask, shrinkage_mask, value_mask, self.all_labels, 
                                    selection=selection_mask if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            self.last_ti_key = key_mask
            self.last_ti_value = value_mask

            if is_deep_update:
                self.memory.set_hidden(hidden_mask)
                self.last_deep_update_ti = self.curr_ti

        # segment the current frame is needed
        if need_segment:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)

            # short term memory 
            batch, num_objects, value_dim, h, w = self.last_ti_value.shape
            last_ti_value = self.last_ti_value.flatten(start_dim=1, end_dim=2)

            if not (msk_ab is not None and not flag_FirstframeIsExemplar):
                memory_value_short, _ = self.network.short_term_attn(key, self.last_ti_key, last_ti_value, None, key.shape[-2:]) 
                memory_value_short = memory_value_short.permute(1, 2, 0).view(batch, num_objects, value_dim, h, w)
                memory_readout += memory_value_short 
            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if msk_ab is not None and flag_FirstframeIsExemplar:
            msk_ab, _ = pad_divide_by(msk_ab, divide_by)
            pred_prob_with_bg = msk_ab

        # save as memory if needed
        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg.unsqueeze(0), is_deep_update=is_deep_update)

            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            self.last_ti_key = key
            self.last_ti_value = value

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)