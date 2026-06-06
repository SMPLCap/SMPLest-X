import argparse
import torch.backends.cudnn as cudnn
from main.config import Config
import os.path as osp
import os
import datetime
from pathlib import Path
import torch.distributed as dist
from utils.distribute_utils import init_distributed_mode, \
    is_main_process, set_seed, get_dist_info
from main.base import Trainer
from human_models.human_models import SMPL, SMPLX
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--local_rank', type=int, dest='num_gpus')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    #parser.add_argument('--master_port', type=int, dest='master_port')
    #parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--config', type=str, default='./config/config_base.py')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    set_seed(2023)
    cudnn.benchmark = True
    
    # process config
    config_path = osp.join('./configs', args.config) # TODO: move config folder outsied main
    cfg = Config.load_config(config_path)
    new_config = {
        "train": {
            "num_gpus": int(args.num_gpus),
        },
        "log":{
            'exp_name':  'test',
            'output_dir': "./output/worldpose_ft",
            'model_dir': "./output/worldpose_ft",
            'log_dir': "./output/worldpose_ft",
            'result_dir': "./output/worldpose_ft",
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    cfg.dump_config()
    # init human models
    smpl = SMPL(cfg.model.human_model_path)
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init traininer
    trainer = Trainer(cfg)
    trainer.logger_info(f"Using {cfg.train.num_gpus} GPUs with bs={cfg.train.train_batch_size} per GPU.")
    trainer.logger_info(f'Training with datasets: {cfg.data.trainset_humandata}')
    
    trainer._make_batch_generator()
    trainer._make_model()
        
    # ddp, align random seed between devices
    #trainer.batch_generator.sampler.set_epoch(0) #reshuffle data each epoch

    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                  record_shapes = True,
                  profile_memory = True) as prof:
        with record_function("model_training:"):
    # with profile(
    #     schedule=schedule(wait=5, warmup=5, active=10, repeat=1),
    #     activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     #on_trace_ready=tensorboard_trace_handler('./outputs/profile'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False,) as prof:
            for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):

                # forward
                trainer.optimizer.zero_grad()
                loss= trainer.model(inputs, targets, meta_info, 'train') #trainer.model.forward(inputs, targets, meta_info, 'train')
                loss_mean = {k: v.mean() for k, v in loss.items()}
                loss_sum = sum(v for k, v in loss_mean.items())
                
                # backward
                loss_sum.backward()
                trainer.optimizer.step()
                trainer.scheduler.step()

                prof.step()
                if itr >= 25:  
                    break

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    main()