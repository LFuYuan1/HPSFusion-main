# HPSFusion-main
This repository provides the official PyTorch implementation of HPSFusion.Text-Guided Hierarchical Perception and Synergistic Interaction Network for Image Fusion.
# Abstract
Infrared and visible image fusion aims to integrate the saliency information from the infrared modality with the rich structural and textural details from the visible modality, thereby generating fused images with more comprehensive and informative representations. However, due to the inherent heterogeneity between the two modalities, existing fusion methods still suffer from limitations in feature alignment stability, frequency-domain consistency, and semantic constraint capability. To address these challenges, we propose a text-guided infrared and visible image fusion framework, termed HPSFusion. Built upon an auto-encoder architecture, HPSFusion employs a hierarchical perception encoder to independently extract modality-specific features, enabling the preservation of both low-level structural details and high-level semantic information. A dual-branch Fourier residual calibration module is introduced to explicitly modulate the frequency-domain responses of different modalities, while a text-guided synergistic interaction module further enhances the semantic consistency of the fused results. Extensive experiments and comprehensive comparisons conducted on the LLVIP, MSRS, and RoadScene datasets demonstrate that HPSFusion consistently outperforms existing state-of-the-art methods in terms of both subjective visual quality and multiple objective evaluation metrics, including EN, SD, and SF.
# Recommended Environment
- [ ] python 3.10.19
- [ ] torch 2.1.1+cu118 
- [ ] torchvision 0.16.1+cu118
- [ ] pandas 2.3.3
- [ ] opencv-python 4.9.0.80
- [ ] numpy 1.24.4
- [ ] pandas 2.3.3
# Training
## 1. Prepare Dataset
Supported datasets include (but are not limited to):
- [ ] MSRS
- [ ] LLVIP
- [ ] RoadScene

Place the dataset in the `'dataset/'` directory.
## 2. Train the Model
Run `python train.py` to train the model.
# Testing
Please place the pretrained weights in the `'pretrained_weights/'` directory.

Run Fusion Testing：`python test.py`
