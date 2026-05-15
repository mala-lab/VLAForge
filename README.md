# VLAForge (CVPR 2026)
Official PyTorch implementation of "Unleashing Vision-Language Semantics for Deepfake Video Detection".

## Overview
Recent Deepfake Video Detection (DFD) studies have demonstrated that pre-trained Vision-Language Models (VLMs) such as CLIP exhibit strong generalization capabilities in detecting artifacts across different identities. However, existing approaches focus on leveraging visual features only, overlooking their most distinctive strength — the rich vision-language semantics embedded in the latent space. We propose VLAForge, a novel DFD framework that unleashes the potential of such cross-modal semantics to enhance model's discriminability in deepfake detection.
This work i) enhances the visual perception of VLM through a `ForgePerceiver`, which acts as an independent learner to capture diverse, subtle forgery cues both granularly and holistically, while preserving the pretrained Vision–Language Alignment (VLA) knowledge, and ii) provides a complementary discriminative cue — `Identity-Aware VLA score`, derived by coupling cross-modal semantics with the forgery cues learned by ForgePerceiver. Notably, the VLA score is augmented by an identity prior-informed text prompting to capture authenticity cues tailored to each identity, thereby enabling more discriminative cross-modal semantics. Comprehensive experiments on video DFD benchmarks, including classical face-swapping forgeries and recent full-face generation forgeries, demonstrate that our VLAForge substantially outperforms state-of-the-art methods at both frame and video levels.

![image](./img/VLAForge.jpg)

## Setup

## Device
- Single NVIDIA GeForce RTX 3090

## Prepare Your Data
#### Step 1. Download the Deepfake Detection Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics), [CDF-v1](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1), [CDF-v2](https://github.com/yuezunli/celeb-deepfakeforensics), [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data), [DeepfakeDetection](https://github.com/ondyari/FaceForensics/tree/master/dataset)

- VQGAN, SiT-XL/2, DiT, PixArt are from [DF40](https://github.com/YZY-stack/DF40) (Celeb-DF).

#### Step 2. The JSON files are provided in [JSONs](https://github.com/mala-lab/VLAForge/tree/main/dataset/dataset_json).

#### Step 3. Download the Pre-train Models on [Google Drive].

## Run VLAForge
#### Quick Start
- Set `test_dataset` to the name of the test dataset in test.ymal. Then, run
```bash
bash test.sh
```

#### Training
- Set `train_dataset` to the name of the test dataset in train.ymal. Then, train your own weights by runing
```bash
bash train.sh
```

## Citation
- If you find the implementation useful, we would appreciate your acknowledgement via citing our VLAForge paper:
```bibtex
@inproceedings{zhu2026dfd,
  title={Unleashing Vision-Language Semantics for Deepfake Video Detection},
  author={Jiawen Zhu, Yunqi Miao, Xueyi Zhang, Jiankang Deng, Guansong Pang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2026}
}
```

