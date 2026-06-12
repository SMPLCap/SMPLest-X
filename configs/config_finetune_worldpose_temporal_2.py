"""Temporal fine-tuning config: freeze ViT-Huge encoder AND decoder, train only the
Temporal Adapter Module (TAM) on WorldPose body data using T=3 stride-3 windows."""

config = {
  "data": {
    "use_cache": False,
    "data_dir": "./data",
    "trainset_humandata": ["WorldPoseTemporal"],
    "testset": "EHF",
    "data_strategy": "concat",
    "total_data_len": "auto",
    "bbox_ratio": 1.2,
    "no_aug": False,
    "WorldPoseTemporal_train_sample_interval": 1,
    "WorldPoseTemporal_test_sample_interval":  1,
  },

  "train": {
    "num_gpus": 1,
    "continue_train": True,
    "start_over": True,          # only the TAM trains -> fresh optimizer / epoch counter
    "end_epoch": 30,
    "train_batch_size": 4,       # encoder runs B*T forwards (under no_grad); lower if OOM
    "num_thread": 1,
    "lr": 1e-4,                  # adapter trains from scratch -> higher than decoder finetune
    "min_lr": 1e-6,
    "save_epoch": 1,
    "save_iters": 1000,          # also save snapshot_latest.pth.tar every 1000 iters
    "remove_checkpoint": False,
    "print_iters": 100,
    # Loss scope: body pose + root orient + shape + body 2D/3D joints. Hand/face off.
    "smplx_kps_3d_weight":   10.0,
    "smplx_kps_2d_weight":    5.0,
    "smplx_pose_weight":     5.0,
    "smplx_shape_weight":     1.0,   # ON (WorldPose stores betas as smplx -> shape valid)
    "smplx_orient_weight":    1.0,
    "hand_root_weight":       0.0,
    "hand_consist_weight":    0.0,
    # "hand_loss":              False,
    # "no_chain_hand_loss":     True,
    "freeze_encoder":         True,
    "freeze_decoder":         True,
  },

  "inference": {
    "num_gpus": 1,
    "detection": {
      "model_type": "yolo",
      "model_path": "./pretrained_models/yolov8x.pt",
      "conf": 0.5,
      "save": False,
      "verbose": False,
      "iou_thr": 0.5,
    },
  },

  "test": {
    "test_batch_size": 1,
  },

  "model": {
    "model_type": "vit_huge",
    "pretrained_model_path": "./pretrained_models/smplest_x_h/smplest_x_h.pth.tar",
    "human_model_path": "./human_models/human_model_files",
    "encoder_pretrained_model_path": "./pretrained_models/vitpose-h.pth",
    "use_temporal": True,
    "temporal": {
      "num_frames":     3,
      "temporal_stride": 3,
      "bottleneck_dim": 256,
      "num_heads":      4,
      # Train on every Nth frame as a center (1 = every frame). Higher = fewer iterations
      # per epoch / faster epochs. Neighbor frames at +/-temporal_stride are still pulled
      # from the full index, so temporal context is preserved regardless of this value.
      "center_frame_interval": 7,
    },
    "encoder_config": {
      "num_classes": 80,
      "task_tokens_num": 80,
      "img_size": (256, 192),
      "patch_size": 16,
      "embed_dim": 1280,
      "depth": 32,
      "num_heads": 16,
      "ratio": 1,
      "use_checkpoint": False,
      "mlp_ratio": 4,
      "qkv_bias": True,
      "drop_path_rate": 0.55,
    },
    "decoder_config": {
      "feat_dim": 1280,
      "dim_out": 512,
      "task_tokens_num": 80,
    },
    "input_img_shape":  (512, 384),
    "input_body_shape": (256, 192),
    "output_hm_shape":  (16, 16, 12),
    "focal":   (5000, 5000),
    "princpt": (192 / 2, 256 / 2),
    "body_3d_size":   2,
    "hand_3d_size":   0.3,
    "face_3d_size":   0.3,
    "camera_3d_size": 2.5,
  },

  "log": {
    "exp_name": None,
    "output_dir": None,
    "model_dir": None,
    "log_dir": None,
    "result_dir": None,
  },
}
