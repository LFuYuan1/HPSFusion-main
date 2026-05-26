# HFSFusion-main
This repository provides the official PyTorch implementation of HFSFusion.A Hierarchical-Frequency Representation Enhancement and Semantic Collaboration Method for Infrared and Visible Image Fusion.
# Abstract
Existing infrared and visible image fusion methods are still hard to balance target saliency, structural detail preservation, and semantic consistency in complex scenes. To address this problem, we propose HFSFusion, an image fusion method based on hierarchical-frequency representation enhancement and semantic collaboration, to improve the overall representation ability of fused images. The network adopts a progressive fusion strategy of hierarchical perception, frequency calibration, and semantic collaboration. Specifically, the hierarchical perception encoder and the dual-branch Fourier residual calibration module enhance hierarchical representations and modulate frequency responses, respectively, thus improving the representation stability of modality features. The text-guided collaborative interaction module guides the network to focus on key target regions and discriminative features. Experiments on the LLVIP, MSRS, and RoadScene datasets show that HFSFusion achieves better overall performance than mainstream methods. In particular, the SSIM scores, which are related to structure preservation, are improved by 9.6%, 8.0%, and 11.9% over the second-best methods, respectively.
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
