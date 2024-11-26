
This repository is the official implementation of [Towards Exact Gradient-based Training on Analog In-memory Computing](https://openreview.net/forum?id=5GwbKlBIIf&referrer=%5Bthe%20profile%20of%20Malte%20J.%20Rasch%5D(%2Fprofile%3Fid%3D~Malte_J._Rasch1)) ([arXiv preprint](https://arxiv.org/abs/2030.12345)). 


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
## Simulation 1
**Figure 1.** Simulation 1 compares digital / analog SGD under different learnable rate 
```bash
python S1-SGD-diff-lr.py
```

## Simulation 2
**Figure 3.** Comparison between digital SGD dynamic, proposed analog SGD dynamic and analog SGD simulated by AIHWkit under different $\tau$ (dynamic range radius)
```bash
python S2-dynamic-verification.py
```

## Simulation 3
**Figure 4.** Ablation study on different parameters, including: 
$\tau$, noise variance, and initialization
```bash
python S3.1-ablation-tau.py
python S3.2-ablation-sigma.py
python S3.3-ablation-init.py
```

## Simulation 4
**Figure 5.** Analog training on MNIST dataset.
The network archetecture can be fully-connected network (FCN) or convolutional neural network (CNN)

Perform simulations on FCN
```bash
python S4.1-mnist-FCN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S4.1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S4.1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.78
python S4.1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.8
python S4.1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S4.1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.78
python S4.1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.8
```
Perform simulations on CNN
```bash
python S4.2-mnist-CNN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S4.2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S4.2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7
python S4.2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.8
python S4.2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S4.2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7
python S4.2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.8
```
After all simulations, the figures can be plotted by
```bash
python S4.1-plot-FCN.py
python S4.2-plot-CNN.py
```

## Simulation 5
**Table 2.** Finetuning Resnet family models on CIFAR10 dataset.
```bash
python A12-resnet-finetune.py --model="Resnet18" -FFT --optimizer="FP SGD" --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet18" -FFT --optimizer="Analog SGD" --tau=0.8 --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet18" -FFT --optimizer="TT-v1" --tau=0.8 --CUDA=${CUDA_IDX} 

python A12-resnet-finetune.py --model="Resnet34" -FFT --optimizer="FP SGD" --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet34" -FFT --optimizer="Analog SGD" --tau=0.8 --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet34" -FFT --optimizer="TT-v1" --tau=0.8 --CUDA=${CUDA_IDX} 

python A12-resnet-finetune.py --model="Resnet50" -FFT --optimizer="FP SGD" --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet50" -FFT --optimizer="Analog SGD" --tau=0.8 --CUDA=${CUDA_IDX} 
python A12-resnet-finetune.py --model="Resnet50" -FFT --optimizer="TT-v1" --tau=0.8 --CUDA=${CUDA_IDX} 
```

## Simulation 6
**Figure 7.** Illustration of weight distribution of FCN model. 
To plot the figure, we need to first save the checkpoint of the models.
```bash
python S4.2-mnist-CNN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX} --save-checkpoint
python S4.1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7 --save-checkpoint
python S4.1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7 --save-checkpoint
```
We could plot the weight distributions after that.
```bash
python S6-plot-distribution.py
```
