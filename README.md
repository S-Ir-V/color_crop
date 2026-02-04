# README
Code for "Dual-branch Aesthetic Image Retouching via Active Reinforcement Learning
for Color Enhancement and Composition Optimization"

# Installation
```shell
# Python 3.8
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow
pip install tensorflow_hub
pip install opencv-python
pip install chainer==7.8.1
pip install chainerrl==0.8.0
pip install gym==0.22.0
pip install cupy-cuda11x==11.6.0
```
# Training
1. Download the Fivek dataset to the specified directory.
2. Download the pre-trained weights to the premodel folder
3. Run train_color.py to start training

# Evaluation
1. Run util/PSNR_SSIM_LIPIPS.py to evaluate metrics
