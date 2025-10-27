
This repository is the official implementation of the following paper:
- [Towards Exact Gradient-based Training on Analog In-memory Computing](https://openreview.net/forum?id=5GwbKlBIIf) (NeurIPS 2024, [ArXiv preprint](https://arxiv.org/abs/2406.12774)). 
- [Analog In-memory Training on General Non-ideal Resistive Elements: The Impact of Response Functions](https://openreview.net/forum?id=tEoHyv61FQ) (NeurIPS 2025, [ArXiv preprint](https://arxiv.org/abs/2502.06309))


# Requirements

This project is built on the analog in-memory computing open-source library [IBM Analog Hardware Acceleration Kit, AIHWKit](https://github.com/IBM/aihwkit).
## Conda installation (Recommanded)
To install requirements:
```bash
conda create -n analog python=3.10
conda activate analog
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge aihwkit-gpu
pip install tensorboard matplotlib numpy
```
## Pip installation
```setup
pip install -r requirements.txt
```

# Running
See the commands in each directories.