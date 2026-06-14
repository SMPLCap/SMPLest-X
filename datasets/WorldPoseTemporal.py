import copy
import numpy as np
import torch

from datasets.WorldPose import WorldPose
from utils.data_utils import load_img, augmentation, generate_patch_image


class WorldPoseTemporal(WorldPose):
    def __init__(self, transform, data_split, cfg):
        super(WorldPoseTemporal, self).__init__(transform, data_split, cfg)

        tcfg = cfg.model.temporal
        self.num_frames = tcfg['num_frames']
        self.stride = tcfg['temporal_stride']
        self.center_idx = self.num_frames // 2
        # frame offsets relative to the center, e.g. [-3, 0, +3] for T=3, stride=3
        self.offsets = [(pos - self.center_idx) * self.stride for pos in range(self.num_frames)]
        self.center_frame_interval = int(tcfg.get('center_frame_interval', 1))

        if len(self.datalist) > 0 and 'person_id' not in self.datalist[0]:
            raise RuntimeError(
                "[WorldPoseTemporal] datalist has no 'person_id'. Regenerate the npz with the "
                "updated scripts/worldpose_to_humandata.py (person identity cannot be recovered "
                "from img_path because multiple persons share a frame).")

        self.index = {}
        for pos, d in enumerate(self.datalist):
            key = (d['seq_name'], d['person_id'])
            self.index.setdefault(key, {})[d['frame_idx']] = pos


        if self.data_split == 'train' and self.center_frame_interval > 1:
            self.center_positions = [
                pos for pos, d in enumerate(self.datalist)
                if d['frame_idx'] % self.center_frame_interval == 0
            ]
        else:
            self.center_positions = list(range(len(self.datalist)))

        print(f"[WorldPoseTemporal] center_frame_interval={self.center_frame_interval} -> "
              f"{len(self.center_positions)} training centers "
              f"(from {len(self.datalist)} total person-frames)")

    def __len__(self):
        if self.data_split == 'train':
            return len(self.center_positions)
        return super(WorldPoseTemporal, self).__len__()

    def _crop_frame(self, data, center_bbox, scale, rot, do_flip, color_scale):
        """Load a neighbor frame and crop it with the CENTER frame's augmentation params."""
        img = load_img(data['img_path'])
        img_patch, _, _ = generate_patch_image(
            img, center_bbox, scale, rot, do_flip, self.cfg.model.input_img_shape)
        img_patch = np.clip(img_patch * color_scale[None, None, :], 0, 255)
        return self.transform(img_patch.astype(np.float32)) / 255.

    def __getitem__(self, idx):
        if self.data_split != 'train':
            return super(WorldPoseTemporal, self).__getitem__(idx)

        real_idx = self.center_positions[idx]
        center = copy.deepcopy(self.datalist[real_idx])
        seq, pid, f0 = center['seq_name'], center['person_id'], center['frame_idx']
        center_bbox = center['bbox']

        img = load_img(center['img_path'])
        no_aug = getattr(self.cfg.data, 'no_aug', False)
        img_aug, img2bb_trans, bb2img_trans, rot, do_flip, scale, color_scale = augmentation(
            no_aug, img, center_bbox, self.data_split, self.cfg.model.input_img_shape)
        center_img = self.transform(img_aug.astype(np.float32)) / 255.

        inputs, targets, meta_info = self._build_train_targets(
            center, center_img, img2bb_trans, bb2img_trans, rot, do_flip, center['img_shape'])

        frame_map = self.index.get((seq, pid), {})
        window_imgs = []
        temporal_valid = []
        for offset in self.offsets:
            if offset == 0:
                window_imgs.append(center_img)
                temporal_valid.append(1.0)
                continue
            pos = frame_map.get(f0 + offset)
            if pos is None:
                window_imgs.append(center_img.clone())
                temporal_valid.append(0.0)
            else:
                neighbor = self.datalist[pos]
                window_imgs.append(
                    self._crop_frame(neighbor, center_bbox, scale, rot, do_flip, color_scale))
                temporal_valid.append(1.0)

        inputs['img'] = torch.stack(window_imgs, dim=0)            
        meta_info['temporal_valid'] = np.array(temporal_valid, dtype=np.float32)  

        return inputs, targets, meta_info
