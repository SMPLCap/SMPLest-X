"""
Verify HumanData .npz by overlaying 2D keypoints and bboxes on images.
Saves overlay images to outputs/verify/

Usage:
    python scripts/verify_humandata.py
"""

import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

NPZ_PATH = Path(r"data/annot/worldpose_val.npz")
# On Windows, remap /mnt/d/... back to D:/...
IMG_ROOT_REMAP = ("/mnt/d/", "/mnt/d/",)
OUT_DIR = Path(r"data/annot/verify")

SEQS_TO_CHECK = ["ARG_CRO_221101"]
FRAMES_PER_SEQ = 10

# SMPL body joint connections for skeleton visualization
SKELETON = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),(20,22),(21,23)
]


def main():
    print(f"Loading {NPZ_PATH}...")
    data = np.load(str(NPZ_PATH), allow_pickle=True)

    image_paths = data['image_path']
    bbox_xywh = data['bbox_xywh']
    kps2d = data['keypoints2d']  # (N, 144, 2)
    smpl = data['smpl'].item()
    meta = data['meta'].item()

    N = len(image_paths)
    print(f"Total samples: {N}")

    # Group samples by sequence and frame
    seq_frames = defaultdict(lambda: defaultdict(list))
    for i in range(N):
        p = image_paths[i]
        parts = p.split('/')
        seq = parts[-2]
        frame_str = parts[-1].replace('.jpg', '')
        if seq in SEQS_TO_CHECK:
            seq_frames[seq][int(frame_str)].append(i)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seq in SEQS_TO_CHECK:
        if seq not in seq_frames:
            print(f"[SKIP] {seq} not found in npz")
            continue

        frames_sorted = sorted(seq_frames[seq].keys())
        total_frames = len(frames_sorted)
        print(f"\n{seq}: {total_frames} frames with annotations")

        # Pick evenly spaced frames
        step = max(1, total_frames // FRAMES_PER_SEQ)
        selected = frames_sorted[::step][:FRAMES_PER_SEQ]

        for frame_idx in selected:
            sample_indices = seq_frames[seq][frame_idx]

            # Load image
            img_path_wsl = image_paths[sample_indices[0]]
            img_path = img_path_wsl.replace(IMG_ROOT_REMAP[0], IMG_ROOT_REMAP[1])
            #img_path = img_path.replace('/', '\\')

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [ERROR] Cannot read {img_path}")
                continue

            # if 'K' in data['meta'].item().keys():
            #     K      = data['meta'].item()['K'][sample_indices[0]]
            #     k_dist = data['meta'].item()['k_dist'][sample_indices[0]]
            #     img = cv2.undistort(img, K, k_dist)


            # Draw each person in this frame
            colors = [
                (0,255,0), (255,0,0), (0,0,255), (255,255,0),
                (255,0,255), (0,255,255), (128,255,0), (255,128,0),
                (0,128,255), (128,0,255), (255,0,128), (0,255,128),
            ]

            for pi, idx in enumerate(sample_indices):
                color = colors[pi % len(colors)]

                # Draw bbox
                bx, by, bw, bh = bbox_xywh[idx][:4]
                x1, y1 = int(bx), int(by)
                x2, y2 = int(bx + bw), int(by + bh)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"P{pi}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw 2D keypoints (first 24 = SMPL body joints)
                joints_2d = kps2d[idx][:24]  # (24, 2)
                for j, (x, y) in enumerate(joints_2d):
                    if x > 0 or y > 0:
                        cv2.circle(img, (int(x), int(y)), 4, color, -1)

                # Draw skeleton
                for (a, b) in SKELETON:
                    if a < 24 and b < 24:
                        xa, ya = joints_2d[a]
                        xb, yb = joints_2d[b]
                        if (xa > 0 or ya > 0) and (xb > 0 or yb > 0):
                            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)

            # Add frame info
            cv2.putText(img, f"{seq} frame={frame_idx:05d} persons={len(sample_indices)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out_path = OUT_DIR / f"{seq}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"  Saved {out_path.name} ({len(sample_indices)} persons)")

    print(f"\nDone. Check results in {OUT_DIR}")


if __name__ == "__main__":
    main()
