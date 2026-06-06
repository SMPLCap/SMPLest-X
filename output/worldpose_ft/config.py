config = {
    "data": {
        "use_cache": false,
        "data_dir": "./data",
        "trainset_humandata": [
            "WorldPose"
        ],
        "testset": "EHF",
        "data_strategy": "concat",
        "total_data_len": "auto",
        "bbox_ratio": 1.2,
        "no_aug": false,
        "WorldPose_train_sample_interval": 1,
        "WorldPose_test_sample_interval": 1
    },
    "train": {
        "num_gpus": 1,
        "continue_train": false,
        "start_over": false,
        "end_epoch": 30,
        "train_batch_size": 16,
        "num_thread": 4,
        "lr": 1e-05,
        "min_lr": 1e-06,
        "save_epoch": 1,
        "remove_checkpoint": false,
        "print_iters": 100,
        "smplx_kps_3d_weight": 10.0,
        "smplx_kps_2d_weight": 3.0,
        "smplx_pose_weight": 5.0,
        "smplx_shape_weight": 1.0,
        "smplx_orient_weight": 5.0,
        "hand_root_weight": 0.0,
        "hand_consist_weight": 0.0,
        "freeze_encoder": true
    },
    "inference": {
        "num_gpus": 1,
        "detection": {
            "model_type": "yolo",
            "model_path": "./pretrained_models/yolov8x.pt",
            "conf": 0.5,
            "save": false,
            "verbose": false,
            "iou_thr": 0.5
        }
    },
    "test": {
        "test_batch_size": 1
    },
    "model": {
        "model_type": "vit_huge",
        "pretrained_model_path": "./outputs/saved_logs/train_worldpose_ft_20260605_105411/model_dump/snapshot_8.pth.tar",
        "human_model_path": "./human_models/human_model_files",
        "encoder_pretrained_model_path": "./pretrained_models/vitpose-h.pth",
        "encoder_config": {
            "num_classes": 80,
            "task_tokens_num": 80,
            "img_size": [
                256,
                192
            ],
            "patch_size": 16,
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "ratio": 1,
            "use_checkpoint": false,
            "mlp_ratio": 4,
            "qkv_bias": true,
            "drop_path_rate": 0.55
        },
        "decoder_config": {
            "feat_dim": 1280,
            "dim_out": 512,
            "task_tokens_num": 80
        },
        "input_img_shape": [
            512,
            384
        ],
        "input_body_shape": [
            256,
            192
        ],
        "output_hm_shape": [
            16,
            16,
            12
        ],
        "focal": [
            5000,
            5000
        ],
        "princpt": [
            96.0,
            128.0
        ],
        "body_3d_size": 2,
        "hand_3d_size": 0.3,
        "face_3d_size": 0.3,
        "camera_3d_size": 2.5
    },
    "log": {
        "exp_name": "test",
        "output_dir": "./output/worldpose_ft",
        "model_dir": "./output/worldpose_ft",
        "log_dir": "./output/worldpose_ft",
        "result_dir": "./output/worldpose_ft"
    }
}
