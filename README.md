# HPSFusion-main
This repository provides the official PyTorch implementation of HPSFusion.Text-Guided Hierarchical Perception and Synergistic Interaction Network for Image Fusion.
# Abstract
Infrared and visible image fusion aims to integrate the salient information provided by infrared modalities with the rich structural and textural details captured by visible images, thereby generating fused images with more informative and comprehensive representations. Owing to the intrinsic heterogeneity between the two modalities, existing fusion methods still encounter significant challenges in feature alignment stability, frequency-domain consistency, and effective semantic constraint modeling. To address these challenges, we propose a text-guided infrared and visible image fusion model, termed HPSFusion. Built upon an autoencoder architecture, HPSFusion employs a hierarchical perceptual encoder to independently extract modality-specific features, enabling the simultaneous preservation of low-level structural details and high-level semantic information. To provide more stable and discriminative feature representations for subsequent fusion, a dual-branch Fourier residual calibration module is designed to explicitly regulate the encoded features from different modalities at the frequency-domain level. Furthermore, to enhance the semantic consistency of the fusion results, a text-guided collaborative interaction module leverages high-level semantic priors to adaptively modulate cross-modal interaction outcomes. Extensive experiments and comprehensive comparative analyses conducted on the LLVIP, MSRS, and RoadScene datasets demonstrate that HPSFusion consistently outperforms existing mainstream methods in terms of both subjective visual quality and multiple objective evaluation metrics.
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
