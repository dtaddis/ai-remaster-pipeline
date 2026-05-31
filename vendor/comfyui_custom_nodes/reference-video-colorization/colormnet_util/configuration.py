from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default

class Configuration():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--no_amp', action='store_true')

        # Data parameters
        parser.add_argument('--davis_root', help='DAVIS and Videvo data root, blackswan/00000.png, blackswan/00001.png, etc', default='')
        parser.add_argument('--validation_root', help='validation data root, blackswan/00000.png, blackswan/00001.png, etc', default='')
        parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

        parser.add_argument('--key_dim', default=64, type=int)
        parser.add_argument('--value_dim', default=512, type=int)
        parser.add_argument('--hidden_dim', default=64, help='Set to =0 to disable', type=int)

        parser.add_argument('--deep_update_prob', default=0.2, type=float)
        parser.add_argument('--stages', help='Training stage (2-DAVIS+Videvo)', default='2')
        parser.add_argument('--savepath', help='dir to save wandb logging files', default='./wandb_save_dir')

        """
        Stage-specific learning parameters
        Batch sizes are effective -- you don't have to scale them when you scale the number processes
        """
        # Stage DAVIS+Videvo
        parser.add_argument('--s2_batch_size', default=2, type=int)
        parser.add_argument('--s2_iterations', default=150000, type=int)
        # fine-tune means fewer augmentations to train the sensory memory
        parser.add_argument('--s2_finetune', default=10000, type=int)
        parser.add_argument('--s2_steps', nargs="*", default=[120000], type=int)

        parser.add_argument('--s2_lr', help='Initial learning rate', default=2e-5, type=float)
        
        parser.add_argument('--s2_num_ref_frames', default=3, type=int)
        parser.add_argument('--s2_num_frames', default=8, type=int)
        parser.add_argument('--s2_start_warm', default=20000, type=int)
        parser.add_argument('--s2_end_warm', default=70000, type=int)

        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.05, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--log_text_interval', default=100, type=int)

        parser.add_argument('--log_image_interval', default=100, type=int)
        
        parser.add_argument('--save_network_interval', default=2500, type=int)
        parser.add_argument('--save_checkpoint_interval', default=5000, type=int)
        parser.add_argument('--exp_id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # # Multiprocessing parameters, not set by users
        # parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['amp'] = not self.args['no_amp']

        # check if the stages are valid
        stage_to_perform = list(self.args['stages'])
        for s in stage_to_perform:
            if s not in ['0', '1', '2', '3']:
                raise NotImplementedError

    def get_stage_parameters(self, stage):
        parameters = {
            'batch_size': self.args['s%s_batch_size'%stage],
            'iterations': self.args['s%s_iterations'%stage],
            'finetune': self.args['s%s_finetune'%stage],
            'steps': self.args['s%s_steps'%stage],
            'lr': self.args['s%s_lr'%stage],
            'num_ref_frames': self.args['s%s_num_ref_frames'%stage],
            'num_frames': self.args['s%s_num_frames'%stage],
            'start_warm': self.args['s%s_start_warm'%stage],
            'end_warm': self.args['s%s_end_warm'%stage],
        }

        return parameters

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
