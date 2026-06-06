"""Fine-tuning config: freeze ViT-Huge encoder, train decoder on WorldPose body data."""

config = {
  "data": {
    "use_cache": False,
    "data_dir": "./data",
    "trainset_humandata": ["WorldPose"],
    "testset": "EHF",
    "data_strategy": "concat",
    "total_data_len": "auto",
    "bbox_ratio": 1.2,
    "no_aug": False,
    "WorldPose_train_sample_interval": 1,
    "WorldPose_test_sample_interval":  1,
  },

  "train": {
    "num_gpus": 1,
    "continue_train":False,     # load pretrained checkpoint
    "start_over": False,         # reset optimizer moments (pretrained moments mislead at new LR)
    "end_epoch": 30,
    "train_batch_size": 16,
    "num_thread": 4,
    "lr": 1e-5,
    "min_lr": 1e-6,
    "save_epoch": 1,
    "remove_checkpoint": False,
    "print_iters": 100,
    # Loss weights — body joints only; hand/face supervision off via validity masks
    # 
    # the 2D head and break the reprojection pipeline's global MPJPE.
    # kps_3d reduced: lower weight minimises collateral shift in shared decoder
    # features that also drive the cam_trans head.
    "smplx_kps_3d_weight":   10.0,
    "smplx_kps_2d_weight":    3.0,
    "smplx_pose_weight":     5.0,
    "smplx_shape_weight":     1.0,
    "smplx_orient_weight":    5.0,
    "hand_root_weight":       0.0,   # no hand supervision
    "hand_consist_weight":    0.0,
    # Freeze encoder — only train decoder
    "freeze_encoder": True,
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
    # Path to the pretrained SMPLest-X checkpoint to fine-tune from
    #"pretrained_model_path":  "./pretrained_models/smplest_x_h/smplest_x_h.pth.tar",
    #"pretrained_model_path":  "./outputs/train_worldpose_ft_20260603_003131/model_dump/snapshot_2.pth.tar",
    "pretrained_model_path":  "./outputs/train_worldpose_ft_20260603_202653/model_dump/snapshot_7.pth.tar",
    "human_model_path": "./human_models/human_model_files",
    "encoder_pretrained_model_path": "./pretrained_models/vitpose-h.pth",
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
