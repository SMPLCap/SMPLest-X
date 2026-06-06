import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import datetime
from tqdm import tqdm
from pathlib import Path


from human_models.human_models import SMPLX

from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Root folder containing one subfolder per sequence.')
    parser.add_argument('--sequences', type=str, default='sequences.txt',
                        help='Text file with one sequence name per line.')
    args = parser.parse_args()
    return args


def load_sequences(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def crop_player(original_img, bbox_xyxy, cfg, transform):
    """Crop and preprocess one bbox. Returns (img_tensor, processed_bbox) or (None, None)."""
    h, w = original_img.shape[:2]
    bbox_xywh = np.array([
        bbox_xyxy[0],
        bbox_xyxy[1],
        abs(bbox_xyxy[2] - bbox_xyxy[0]),
        abs(bbox_xyxy[3] - bbox_xyxy[1]),
    ])
    bbox = process_bbox(bbox=bbox_xywh, img_width=w, img_height=h,
                        input_img_shape=cfg.model.input_img_shape,
                        ratio=getattr(cfg.data, "bbox_ratio", 1.25))
    if bbox is None:
        return None, None
    img, _, _ = generate_patch_image(cvimg=original_img, bbox=bbox,
                                     scale=1.0, rot=0.0, do_flip=False,
                                     out_shape=cfg.model.input_img_shape)
    img = transform(img.astype(np.float32)) / 255  # (3, H, W)
    return img, bbox


def run_batch(demoer, imgs):
    """Run model on a batch of pre-cropped tensors. imgs: (B, 3, H, W) cuda tensor."""
    with torch.no_grad():
        out = demoer.model({'img': imgs}, {}, {}, 'test')
    return {
        'global_orient': out['smplx_root_pose'].detach().cpu().numpy(),  # (B, 3)
        'body_pose':     out['smplx_body_pose'].detach().cpu().numpy(),  # (B, 63)
        'betas':         out['smplx_shape'].detach().cpu().numpy(),      # (B, 10)
        'transl':        out['cam_trans'].detach().cpu().numpy(),        # (B, 3)
    }


def process_sequence(demoer, cfg, root_dir, file_name, boxes_path, img_folder):
    """Run inference on one sequence and save poses to demo/<file_name>_poses.npz."""

    all_boxes = np.load(boxes_path)  # (T, N, 4)
    assert all_boxes.ndim == 3 and all_boxes.shape[2] == 4, \
        f"Expected boxes shape (T, N, 4), got {all_boxes.shape}"
    T, N, _ = all_boxes.shape  # T frames derived from boxes file

    transform = transforms.ToTensor()

    global_orient = np.full((N, T, 3),  np.nan, dtype=np.float32)
    body_poses     = np.full((N, T, 63), np.nan, dtype=np.float32)
    transl         = np.full((N, T, 3),  np.nan, dtype=np.float32)
    betas_sum      = np.zeros((N, 10),   dtype=np.float32)
    betas_count    = np.zeros(N,         dtype=np.int32)
    bboxes_xywh    = np.full((N, T, 4),  np.nan, dtype=np.float32)

    # Single pass: crop players per frame, run inference immediately (no global accumulation)
    for t, frame_idx in enumerate(tqdm(range(0, T ), desc=f"[{file_name}]")):
        img_path = osp.join(img_folder, f'{frame_idx:05d}.jpg')
        original_img = load_img(img_path)

        visible_ns, imgs, bboxes = [], [], []
        for n in range(N):
            bbox_xyxy = all_boxes[t, n]
            if np.any(np.isnan(bbox_xyxy)):
                continue
            img_tensor, bbox = crop_player(original_img, bbox_xyxy, cfg, transform)
            if bbox is None:
                continue
            visible_ns.append(n)
            imgs.append(img_tensor)
            bboxes.append(bbox)

        if not visible_ns:
            continue

        try:
            batch_tensor = torch.stack(imgs).cuda()
            results = run_batch(demoer, batch_tensor)
        except torch.cuda.OutOfMemoryError:
            # fall back to one player at a time if batch exceeds GPU memory
            torch.cuda.empty_cache()
            results = {'global_orient': [], 'body_pose': [], 'betas': [], 'transl': []}
            for img in imgs:
                r = run_batch(demoer, img.unsqueeze(0).cuda())
                for k in results:
                    results[k].append(r[k][0])
            results = {k: np.stack(v) for k, v in results.items()}

        for i, n in enumerate(visible_ns):
            global_orient[n, t]  = results['global_orient'][i]
            body_poses[n, t]     = results['body_pose'][i]
            transl[n, t]         = results['transl'][i]
            betas_sum[n]        += results['betas'][i]
            betas_count[n]      += 1
            bboxes_xywh[n, t]    = bboxes[i]

    betas = np.where(
        betas_count[:, None] > 0,
        betas_sum / np.maximum(betas_count[:, None], 1),
        np.nan
    ).astype(np.float32)
    
    poses_path = osp.join(root_dir, 'demo', f'{file_name}_poses.npz')
    np.savez(poses_path,
             betas=betas,
             global_orient=global_orient,
             body_poses=body_poses,
             transl=transl,
             bboxes_xywh=bboxes_xywh,
             focal=np.array(cfg.model.focal, dtype=np.float32),
             princpt=np.array(cfg.model.princpt, dtype=np.float32),
             input_body_shape=np.array(cfg.model.input_body_shape, dtype=np.float32))
    print(f"Saved {poses_path}  betas={betas.shape} global_orient={global_orient.shape} "
          f"body_poses={body_poses.shape} transl={transl.shape}")


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"CUDA: {torch.cuda.get_device_name(0)}, {torch.cuda.memory_allocated()/1e9:.1f}GB used")
    cudnn.benchmark = True

    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')

    new_config = {
        "model": {"pretrained_model_path": checkpoint_path},
        "log": {
            'exp_name': f'inference_{args.ckpt_name}_{time_str}',
            'log_dir': osp.join(root_dir, 'outputs', f'inference_{args.ckpt_name}_{time_str}', 'log'),
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    SMPLX(cfg.model.human_model_path)  # init singleton before model creation

    demoer = Tester(cfg)
    demoer.logger.info("Using 1 GPU.")
    demoer._make_model()

    file_names = load_sequences(args.sequences)
    print(f"Loaded {len(file_names)} sequences from {args.sequences}")
    n_seq = len(file_names)
    for i, file_name in enumerate(file_names):
        img_folder = osp.join(args.input_folder, file_name)
        boxes_path = osp.join(root_dir, 'demo', 'boxes', f'{file_name}.npy')
        print(f"\n=== Sequence {i+1}/{n_seq}: {file_name} ===")
        print(f"    frames: {img_folder}")
        print(f"    boxes:  {boxes_path}")
        process_sequence(demoer, cfg, root_dir,
                         file_name=file_name,
                         boxes_path=boxes_path,
                         img_folder=img_folder)

    print("\nAll sequences done.")


if __name__ == "__main__":
    main()
