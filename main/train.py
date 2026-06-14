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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, dest='num_gpus')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--master_port', type=int, dest='master_port')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--config', type=str, default='./config/config_base.py')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    set_seed(2023)
    cudnn.benchmark = True
    
    # process config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./configs', args.config) # TODO: move config folder outsied main
    cfg = Config.load_config(config_path)
    new_config = {
        "train": {
            "num_gpus": int(args.num_gpus),
        },
        "log":{
            'exp_name':  f'{args.exp_name}_{time_str}',
            'output_dir': osp.join(root_dir, 'outputs', f'{args.exp_name}_{time_str}'),
            'model_dir': osp.join(root_dir, 'outputs', f'{args.exp_name}_{time_str}', 'model_dump'),
            'log_dir': osp.join(root_dir, 'outputs', f'{args.exp_name}_{time_str}', 'log'),
            'result_dir': osp.join(root_dir, 'outputs', f'{args.exp_name}_{time_str}', 'result'),
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    cfg.dump_config()

    # init ddp
    distributed, gpu_idx = init_distributed_mode(args.master_port)
    
    # init human models
    smpl = SMPL(cfg.model.human_model_path)
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init traininer
    trainer = Trainer(cfg, distributed, gpu_idx)
    trainer.logger_info(f"Using {cfg.train.num_gpus} GPUs with bs={cfg.train.train_batch_size} per GPU.")
    trainer.logger_info(f'Training with datasets: {cfg.data.trainset_humandata}')
    
    trainer._make_batch_generator()
    trainer._make_model()

    for epoch in range(trainer.start_epoch, cfg.train.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        
        # ddp, align random seed between devices
        trainer.batch_generator.sampler.set_epoch(epoch)

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss= trainer.model(inputs, targets, meta_info, 'train')
            loss_mean = {k: v.mean() for k, v in loss.items()}
            loss_sum = sum(v for k, v in loss_mean.items())
            
            # backward
            loss_sum.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()

            trainer.gpu_timer.toc()

            if (itr + 1) % cfg.train.print_iters == 0:
                # loss of all ranks
                rank, world_size = get_dist_info()
                loss_print = loss_mean.copy()
                for k in loss_print:
                    dist.all_reduce(loss_print[k]) 
                
                total_loss = 0
                for k in loss_print:
                    loss_print[k] = loss_print[k] / world_size
                    total_loss += loss_print[k]
                loss_print['total'] = total_loss
                    
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.train.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss_print.items()]
                trainer.logger_info(' '.join(screen))

            # periodic mid-epoch checkpoint (safety net for long epochs / interruptions)
            save_iters = getattr(cfg.train, 'save_iters', 0)
            if save_iters and is_main_process() and (itr + 1) % save_iters == 0:
                trainer.save_model({
                    'epoch': epoch,
                    'network': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                }, epoch, name='snapshot_latest.pth.tar')

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save model ddp, save model.module on rank 0 only
        save_epoch = getattr(cfg.train, 'save_epoch', 5)
        previous_saved_epoch = None
        remove_previous = getattr(cfg.train, 'remove_checkpoint', False)
        if is_main_process() and (epoch % save_epoch == 0 or epoch == cfg.train.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

            # remove previous
            if previous_saved_epoch is not None and remove_previous:
                to_remove = osp.join(cfg.log.model_dir, f'snapshot_{str(previous_saved_epoch)}.pth.tar')
                os.remove(to_remove)
                previous_saved_epoch = epoch

        dist.barrier()

if __name__ == "__main__":
    main()