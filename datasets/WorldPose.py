import os.path as osp
from datasets.humandata import HumanDataset


class WorldPose(HumanDataset):
    def __init__(self, transform, data_split, cfg):
        super(WorldPose, self).__init__(transform, data_split, cfg)

        self.cfg = cfg
        self.use_cache = getattr(self.cfg.data, 'use_cache', False)
        self.annot_path_cache = osp.join(self.cfg.data.data_dir, 'cache', f'worldpose_{self.data_split}.npz')

        self.img_shape = None
        self.cam_param = {}

        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] Loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            self.datalist = []
            self.img_dir = ''

            if self.data_split == 'train':
                filename = osp.join('train', 'worldpose_train.npz')
            else:
                filename = osp.join('train', 'worldpose_train.npz')

            self.annot_path = osp.join(self.cfg.data.data_dir, filename)

            self.datalist = self.load_data(
                train_sample_interval=getattr(self.cfg.data, f'{self.__class__.__name__}_train_sample_interval', 1),
                test_sample_interval=getattr(self.cfg.data, f'{self.__class__.__name__}_test_sample_interval', 10))

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
