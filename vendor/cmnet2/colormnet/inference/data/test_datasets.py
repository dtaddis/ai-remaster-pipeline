import os
from os import path
import json

from inference.data.video_reader import VideoReader_221128_TransColorization


class DAVISTestDataset_221128_TransColorization_batch:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        self.image_dir = data_root
        self.mask_dir = imset
        self.size_dir = data_root
        self.size = size

        self.vid_list = [clip_name for clip_name in sorted(os.listdir(data_root)) if
                         clip_name != '.DS_Store' and not clip_name.startswith('.')]

        # print(lst, len(lst), self.vid_list, self.vid_list_DAVIS2016, path.join(data_root, 'ImageSets', imset));assert 1==0

    def get_datasets(self):
        for video in self.vid_list:
            # print(self.image_dir, video, path.join(self.image_dir, video));assert 1==0
            yield VideoReader_221128_TransColorization(vid_name=video,
                                                       image_dir=path.join(self.image_dir, video),
                                                       mask_dir=path.join(self.mask_dir, video),
                                                       size=self.size,
                                                       to_save=None,
                                                       use_all_mask=True,
                                                       size_dir=path.join(self.size_dir, video),
                                                       )

    def __len__(self):
        return len(self.vid_list)
