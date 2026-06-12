"""
predict_worldpose_seq.py

Run a trained temporal (TAM) SMPLest-X checkpoint over a full WorldPose sequence and save
the predicted SMPL-X parameters per (frame, person). Works on held-out sequences (e.g.
ARG_CRO_220001) because it reads the RAW GT boxes directly — the column index in the
boxes array is the persistent person track, so T=3 stride-3 temporal windows are built
per-player without a detector.

Outputs:
  outputs/predictions/<seq>_<ckpt>.npz   (predicted SMPL-X params + frame/person ids)
  outputs/predictions/<seq>/<frame>.jpg  (rendered overlay, if --render)

Usage (WSL, smplestx env, repo root):
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  PYTHONPATH=.:$PYTHONPATH python scripts/predict_worldpose_seq.py \
      --ckpt_path outputs/train_worldpose_temporal_20260601_135240/model_dump/snapshot_1.pth.tar \
      --seq ARG_CRO_220001 --start 0 --end 1231 --frame_stride 1 --render
"""

import os
import os.path as osp
import sys
import argparse
import datetime
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, osp.join(osp.dirname(__file__)))  # for scripts/config.py
from config import WP_ROOT, WP_PATHS

from human_models.human_models import SMPLX
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--config', type=str, default='config_finetune_worldpose_temporal.py')
    p.add_argument('--seq', type=str, default='ARG_CRO_220001')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=None, help='inclusive last frame (default: last)')
    p.add_argument('--frame_stride', type=int, default=1, help='process every Nth frame')
    p.add_argument('--max_persons', type=int, default=99)
    p.add_argument('--render', action='store_true', help='also save rendered overlay frames')
    return p.parse_args()


def main():
    args = parse_args()
    cudnn.benchmark = True
    root_dir = Path(__file__).resolve().parent.parent
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ---- model (temporal config -> builds TAM) + trained checkpoint ----
    cfg = Config.load_config(osp.join('./configs', args.config))
    exp_name = f'predict_{args.seq}_{time_str}'
    cfg.update_config({
        "model": {"pretrained_model_path": args.ckpt_path},
        "log": {"exp_name": exp_name, "log_dir": osp.join(root_dir, 'outputs', exp_name, 'log')},
    })
    cfg.prepare_log()

    tcfg = cfg.model.temporal
    num_frames, stride = tcfg['num_frames'], tcfg['temporal_stride']
    center_idx = num_frames // 2
    offsets = [(pos - center_idx) * stride for pos in range(num_frames)]

    smpl_x = SMPLX(cfg.model.human_model_path)
    demoer = Tester(cfg)
    demoer._make_model()

    # ---- raw GT boxes (T, N, 4) xyxy ; column = person track ----
    boxes = np.load(WP_PATHS["boxes"] / f"{args.seq}.npy")   # (T, N, 4)
    T, N = boxes.shape[:2]
    img_dir = WP_ROOT / "images" / args.seq
    end = args.end if args.end is not None else T - 1
    frames = list(range(args.start, min(end, T - 1) + 1, args.frame_stride))
    print(f"[{args.seq}] frames {args.start}..{end} (stride {args.frame_stride}) | "
          f"{N} tracks | window offsets {offsets}")

    transform = transforms.ToTensor()
    img_cache = {}

    def get_img(f):
        if f not in img_cache:
            fp = str(img_dir / f"{f:05d}.jpg")
            img_cache[f] = load_img(fp) if osp.isfile(fp) else None
        return img_cache[f]

    def valid_box(f, p):
        if f < 0 or f >= T:
            return None
        b = boxes[f, p]
        if np.isnan(b).any():
            return None
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 5 or h < 5:
            return None
        return np.array([b[0], b[1], w, h], dtype=np.float32)  # xywh

    out_dir = osp.join(root_dir, 'outputs', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    if args.render:
        os.makedirs(osp.join(out_dir, args.seq), exist_ok=True)

    records = []  # one dict per (frame, person)

    for f in frames:
        center_img = get_img(f)
        if center_img is None:
            continue
        Hh, Ww = center_img.shape[:2]
        vis = center_img.copy() if args.render else None

        for p in range(min(N, args.max_persons)):
            bw = valid_box(f, p)
            if bw is None:
                continue
            bbox = process_bbox(bbox=bw.copy(), img_width=Ww, img_height=Hh,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))
            if bbox is None:
                continue

            center_crop, _, _ = generate_patch_image(center_img, bbox, 1.0, 0.0, False,
                                                     cfg.model.input_img_shape)
            center_t = transform(center_crop.astype(np.float32)) / 255

            window, tvalid = [], []
            for off in offsets:
                if off == 0:
                    window.append(center_t); tvalid.append(1.0); continue
                nf = f + off
                nb = valid_box(nf, p)
                nimg = get_img(nf) if nb is not None else None
                if nb is None or nimg is None:
                    window.append(center_t.clone()); tvalid.append(0.0)
                else:
                    crop, _, _ = generate_patch_image(nimg, bbox, 1.0, 0.0, False,
                                                      cfg.model.input_img_shape)
                    window.append(transform(crop.astype(np.float32)) / 255)
                    tvalid.append(1.0)

            imgs = torch.stack(window, dim=0)[None].cuda()                 # (1,T,3,H,W)
            meta_info = {'temporal_valid': torch.tensor(tvalid, dtype=torch.float32)[None].cuda()}
            with torch.no_grad():
                out = demoer.model({'img': imgs}, {}, meta_info, 'test')

            records.append({
                'frame': f, 'person_id': p,
                'root_pose':  out['smplx_root_pose'][0].cpu().numpy(),    # (3,) axis-angle
                'body_pose':  out['smplx_body_pose'][0].cpu().numpy(),    # (63,)
                'lhand_pose': out['smplx_lhand_pose'][0].cpu().numpy(),   # (45,)
                'rhand_pose': out['smplx_rhand_pose'][0].cpu().numpy(),   # (45,)
                'jaw_pose':   out['smplx_jaw_pose'][0].cpu().numpy(),     # (3,)
                'betas':      out['smplx_shape'][0].cpu().numpy(),        # (10,)
                'expr':       out['smplx_expr'][0].cpu().numpy(),         # (10,)
                'cam_trans':  out['cam_trans'][0].cpu().numpy(),          # (3,)
                'bbox_xywh':  bbox.astype(np.float32),                    # PROCESSED bbox that built the crop (defines the virtual crop camera), not raw bw
            })

            if args.render:
                mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
                focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                         cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
                princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                           cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
                vis = render_mesh(vis, mesh, smpl_x.face,
                                  {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

        if args.render and vis is not None:
            cv2.imwrite(osp.join(out_dir, args.seq, f"{f:05d}.jpg"), vis[:, :, ::-1])

        # free old frames from cache to bound memory
        for key in [k for k in img_cache if k < f - max(offsets) - 1]:
            del img_cache[key]

        if f % 50 == 0:
            print(f"  frame {f}/{end}  records={len(records)}")

    # ---- save predictions ----
    ckpt_tag = osp.splitext(osp.basename(args.ckpt_path))[0]
    out_npz = osp.join(out_dir, f"{args.seq}_{ckpt_tag}.npz")
    keys = ['frame', 'person_id', 'root_pose', 'body_pose', 'lhand_pose', 'rhand_pose',
            'jaw_pose', 'betas', 'expr', 'cam_trans', 'bbox_xywh']
    packed = {k: np.array([r[k] for r in records]) for k in keys}
    np.savez_compressed(out_npz, **packed)
    print(f"\nSaved {len(records)} predictions -> {out_npz}")
    if args.render:
        print(f"Rendered frames -> {osp.join(out_dir, args.seq)}/")


if __name__ == '__main__':
    main()
