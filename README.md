# SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation

This work is the extended version of [SMPLer-X](https://arxiv.org/abs/2309.17448). This new codebase is designed for easy installation and flexible development, enabling seamless integration of new methods with the pretrained SMPLest-X model.

![Teaser](./assets/teaser.png)


## Useful links

<div align="center">
    <a href="https://arxiv.org/abs/2501.09782" class="button"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://caizhongang.github.io/projects/SMPLer-X/" class="button"><b>[Homepage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://youtu.be/DepTqbPpVzY" class="button"><b>[Video]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/caizhongang/SMPLer-X" class="button"><b>[SMPLer-X]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/open-mmlab/mmhuman3d" class="button"><b>[MMHuman3D]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/wqyin/WHAC/tree/main" class="button"><b>[WHAC]</b></a></a>
    
</div>


## News

- [2025-10-21] SMPLest-X accepted to TPAMI.
- [2025-02-17] Pretrained model available for download.
- [2025-02-14] 💌💌💌 Brand new codebase released for training, testing and inference.
- [2025-01-20] Paper released on [arXiv](https://arxiv.org/abs/2501.09782).
- [2025-01-08] Project page created.


## Install
```bash
bash scripts/install.sh
```

## Preparation

#### SMPLest-X pretrained models
- Download the pretrained **SMPLest-X-Huge model** weight from [here](https://huggingface.co/waanqii/SMPLest-X/tree/main) (8.2G).
- Place the pretrained weight and respective config file according to the file structure.

#### Parametric human models
- Download [SMPL-X](https://smpl-x.is.tue.mpg.de/) and [SMPL](https://smpl.is.tue.mpg.de/) body models.

#### ViT-Pose pretrained models (For training only)
- Follow [OSX](https://github.com/IDEA-Research/OSX) in preparing pretrained ViTPose models. Download the ViTPose pretrained weights from [here](https://github.com/ViTAE-Transformer/ViTPose).

#### HumanData
- Please refer to [this guide](humandata_prep/README.md) for instructions on preparing the data in the HumanData format.

The final file structure should be like:
```
.
├── assets
├── configs
├── data
│   ├── annot # humandata.npz files
│   ├── cache # cached humandata
│   └── img # original data files
├── datasets
├── demo
├── human_models
│   └── human_model_files # parametric human models
├── main
├── models
├── outputs
│   └── smplest_x_h
├── pretrained_models
│   ├── vitpose_huge.pth # for training only
│   ├── yolov8x.pt # auto download during inference
│   └── smplest_x_h
│       ├── smplest_x_h.pth.tar
│       └── config_base.py
├── scripts
├── utils
├── README.md
└── requirements.txt
```

## Inference 

- Place the video for inference under `SMPLest-X/demo`
- Prepare the pretrained model under `SMPLest-X/pretrained_models`
- Pretrained YOLO model will be downloaded automatically during the first time usage.
- Inference output will be saved in `SMPLest-X/demo`

```bash
sh scripts/inference.sh {MODEL_DIR} {FILE_NAME} {FPS}

# For inferencing test_video.mp4 (30FPS) with SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar
sh scripts/inference.sh i will run inferencesmplest_x_h test_video.mp4 30
```


## Training
```bash
bash scripts/train.sh {JOB_NAME} {NUM_GPUS} {CONFIG_FILE}

# For training SMPLest-X-H with 16 GPUS
bash scripts/train.sh smplest_x_h 16 config_smplest_x_h.py
```
- CONFIG_FILE is the file name under `SMPLest-X/config`
- Logs and checkpoints will be saved to `SMPLest-X/outputs/train_{JOB_NAME}_{DATE_TIME}`


## Testing
```bash
sh scripts/test.sh {TEST_DATSET} {MODEL_DIR} {CKPT_ID}

# For testing the model SMPLest-X/outputs/smplest_x_h/model_dump/snapshot_5.pth.tar 
# on dataset SynHand
sh scripts/test.sh SynHand smplest_x_h 5
```
- NUM_GPU = 1 is used by default for testing
- Logs and results  will be saved to `SMPLest-X/outputs/test_{TEST_DATSET}_ep{CKPT_ID}_{DATE_TIME}`


## FAQ
- How do I animate my virtual characters with SMPLest-X output (like that in the demo video)? 
  - We are working on that, please stay tuned!
    Currently, this repo supports SMPL-X estimation and a simple visualization (overlay of SMPL-X vertices).


## Citation
```text
# SMPLest-X
@article{yin2025smplest,
    title={SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation}, 
    author={Yin, Wanqi and Cai, Zhongang and Wang, Ruisi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Mei, Haiyi and Wang, Yanjun and Pang, Hui En and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yamashita, Atsushi and Yang, Lei and Liu, Ziwei},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    year={2026},
    volume={48},
    number={2},
    pages={1778-1794},
    doi={10.1109/TPAMI.2025.3618174}
}

# SMPLer-X
@inproceedings{cai2023smplerx,
    title={{SMPLer-X}: Scaling up expressive human pose and shape estimation},
    author={Cai, Zhongang and Yin, Wanqi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Yanjun, Wang and Pang, Hui En and Mei, Haiyi and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yang, Lei and Liu, Ziwei},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023}
}
```

## Explore More [Motrix](https://github.com/MotrixLab) Projects

### Motion Capture
- [SMPL-X] [TPAMI'25] [SMPLest-X](https://github.com/MotrixLab/SMPLest-X): An extended version of [SMPLer-X](https://github.com/MotrixLab/SMPLer-X) with stronger foundation models.
- [SMPL-X] [NeurIPS'23] [SMPLer-X](https://github.com/MotrixLab/SMPLer-X): Scaling up EHPS towards a family of generalist foundation models.
- [SMPL-X] [ECCV'24] [WHAC](https://github.com/MotrixLab/WHAC): World-grounded human pose and camera estimation from monocular videos.
- [SMPL-X] [CVPR'24] [AiOS](https://github.com/MotrixLab/AiOS): An all-in-one-stage pipeline combining detection and 3D human reconstruction. 
- [SMPL-X] [NeurIPS'23] [RoboSMPLX](https://github.com/MotrixLab/RoboSMPLX): A framework to enhance the robustness of whole-body pose and shape estimation.
- [SMPL-X] [ICML'25] [ADHMR](https://github.com/MotrixLab/ADHMR): A framework to align diffusion-based human mesh recovery methods via direct preference optimization.
- [SMPL-X] [MKA](https://github.com/MotrixLab/MKA): Full-body 3D mesh reconstruction from single- or multi-view RGB videos.
- [SMPL] [ICCV'23] [Zolly](https://github.com/MotrixLab/Zolly): 3D human mesh reconstruction from perspective-distorted images.
- [SMPL] [IJCV'26] [PointHPS](https://github.com/MotrixLab/PointHPS): 3D HPS from point clouds captured in real-world settings.
- [SMPL] [NeurIPS'22] [HMR-Benchmarks](https://github.com/MotrixLab/hmr-benchmarks): A comprehensive benchmark of HPS datasets, backbones, and training strategies.

### Motion Generation
- [SMPL-X] [ICLR'26] [ViMoGen](https://github.com/MotrixLab/ViMoGen): A comprehensive framework that transfers knowledge from ViGen to MoGen across data, modeling, and evaluation.
- [SMPL-X] [ECCV'24] [LMM](https://github.com/MotrixLab/LMM): Large Motion Model for Unified Multi-Modal Motion Generation.
- [SMPL-X] [NeurIPS'23] [FineMoGen](https://github.com/MotrixLab/FineMoGen): Fine-Grained Spatio-Temporal Motion Generation and Editing.
- [SMPL] [InfiniteDance](https://github.com/MotrixLab/InfiniteDance): A large-scale 3D dance dataset and an MLLM-based music-to-dance model designed for robust in-the-wild generalization.
- [SMPL] [NeurIPS'23] [InsActor](https://github.com/MotrixLab/insactor): Generating physics-based human motions from language and waypoint conditions via diffusion policies.
- [SMPL] [ICCV'23] [ReMoDiffuse](https://github.com/MotrixLab/ReMoDiffuse): Retrieval-Augmented Motion Diffusion Model.
- [SMPL] [TPAMI'24] [MotionDiffuse](https://github.com/MotrixLab/MotionDiffuse): Text-Driven Human Motion Generation with Diffusion Model.

### Motion Dataset
- [SMPL] [ECCV'22] [HuMMan](https://github.com/MotrixLab/humman_toolbox): Toolbox for HuMMan, a large-scale multi-modal 4D human dataset.
- [SMPLX] [T-PAMI'24] [GTA-Human](https://github.com/MotrixLab/gta-human_toolbox): Toolbox for GTA-Human, a large-scale 3D human dataset generated with the GTA-V game engine.
