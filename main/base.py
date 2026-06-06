import os.path as osp
import math
import abc
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim
import torchvision.transforms as transforms
from utils.timer import Timer
from utils.logger import colorlogger
from datasets.dataset import MultipleDatasets
import importlib
from models.SMPLest_X import get_model

# ddp
import torch.cuda
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.utils.data.distributed
from utils.distribute_utils import (
    get_rank, is_main_process, time_synchronized, get_group_idx, get_process_groups, get_dist_info
)

def dynamic_import(module_name, object_name):
    """Dynamically import a module and access a specific object."""
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self, cfg, distributed=False, gpu_idx=None):
        super(Trainer, self).__init__(cfg, log_name='train_logs.txt')
        self.distributed = distributed
        self.gpu_idx = gpu_idx
        self.cfg = cfg

    def get_optimizer(self, model):
        normal_param = []

        freeze_encoder = getattr(self.cfg.train, 'freeze_encoder', False)
        if freeze_encoder:
            # Freeze encoder weights and exclude from optimizer
            for param in model.module.encoder.parameters():
                param.requires_grad = False
            self.logger.info('Encoder frozen — only training decoder.')
            modules_to_train = [m for m in model.module.trainable_modules
                                if m is not model.module.encoder]
        else:
            modules_to_train = model.module.trainable_modules

        for module in modules_to_train:
            normal_param += list(module.parameters())
        optim_params = [
            {
                'params': normal_param,
                'lr': self.cfg.train.lr
            }
        ]
        optimizer = torch.optim.Adam(optim_params, lr=self.cfg.train.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.log.model_dir, f'snapshot_{str(epoch)}.pth.tar')

        # do not save smplx layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smplx_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info(f"Write snapshot into {file_path}")

    def load_model(self, model, optimizer):
        if self.cfg.model.pretrained_model_path is not None:
            ckpt_path = self.cfg.model.pretrained_model_path
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) # solve CUDA OOM error in DDP
            model.load_state_dict(ckpt['network'], strict=False)
            model.cuda()
            self.logger.info(f'Load checkpoint from {ckpt_path}')
            torch.cuda.empty_cache()
            if getattr(self.cfg.train, 'start_over', True):
                start_epoch = 0
            else:
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1 
                self.logger.info(f'Load optimizer, start from {start_epoch}')
        else:
            start_epoch = 0

        return start_epoch, model, optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger_info("Creating dataset...")
        trainset_humandata_loader = []
        for humandata_dataset in self.cfg.data.trainset_humandata:
            trainset_humandata_loader.append(dynamic_import(
                f"datasets.{humandata_dataset}", humandata_dataset)(transforms.ToTensor(), "train", self.cfg))
        
        data_strategy = getattr(self.cfg.data, 'data_strategy', 'balance')
        if data_strategy == 'concat':
            print("Using [concat] strategy...")
            trainset_loader = MultipleDatasets(trainset_humandata_loader, 
                                                make_same_len=False, verbose=True)
        elif data_strategy == 'balance':
            total_len = getattr(self.cfg.data, 'total_data_len', 'auto')
            print(f"Using [balance] strategy with total_data_len : {total_len}...")
            trainset_loader = MultipleDatasets(trainset_humandata_loader, 
                                                 make_same_len=True, total_len=total_len, verbose=True)
      
        self.itr_per_epoch = math.ceil(
            len(trainset_loader) / self.cfg.train.num_gpus / self.cfg.train.train_batch_size)

        if self.distributed:
            self.logger_info(f"Total data length {len(trainset_loader)}.")
            rank, world_size = get_dist_info()
            self.logger_info("Using distributed data sampler.")
            
            sampler_train = DistributedSampler(trainset_loader, world_size, rank, shuffle=True)
            self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.train.train_batch_size,
                                            shuffle=False, num_workers=self.cfg.train.num_thread, sampler=sampler_train,
                                            pin_memory=True, persistent_workers=True if self.cfg.train.num_thread > 0 else False, 
                                            drop_last=True)
        else:
            self.batch_generator = DataLoader(dataset=trainset_loader, 
                                              batch_size=self.cfg.train.num_gpus * self.cfg.train.train_batch_size,
                                              shuffle=True, num_workers=self.cfg.train.num_thread,
                                              pin_memory=True, drop_last=True)

    def _make_model(self):
        # prepare network
        self.logger_info("Creating graph and optimizer...")
        model = get_model(self.cfg, 'train')
        
        if self.distributed:
            self.logger_info("Using distributed data parallel.")
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.gpu_idx],
                find_unused_parameters=True) 
        else:
            model = DataParallel(model).cuda()

        optimizer = self.get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        self.cfg.train.end_epoch * self.itr_per_epoch,
                                        eta_min=getattr(self.cfg.train,'min_lr',1e-6))

        if self.cfg.train.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def logger_info(self, info):
        if self.distributed:
            if is_main_process():
                self.logger.info(info)
        else:
            self.logger.info(info)


class Tester(Base):
    def __init__(self, cfg):
        super(Tester, self).__init__(cfg, log_name='test_logs.txt')

        self.cfg = cfg

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = dynamic_import(
                f"datasets.{self.cfg.data.testset}", self.cfg.data.testset)(transforms.ToTensor(), "test", self.cfg)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.test.test_batch_size,
                                     shuffle=False, num_workers=1, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        self.logger.info('Load checkpoint from {}'.format(self.cfg.model.pretrained_model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.cfg, 'test') #smplestx model function
        model = DataParallel(model).cuda()

        ckpt = torch.load(self.cfg.model.pretrained_model_path, map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            if 'module' not in k:
                k = 'module.' + k
            k = k.replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
                'hand_rotation_net', 'hand_regressor')
            new_state_dict[k] = v
        self.logger.warning("Attention: Strict=False is set for checkpoint loading. Please check manually.")
        model.load_state_dict(new_state_dict, strict=False)
        model.cuda()
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)
