# Running
In the bash following commands, replace the `${CUDA_IDX}` variable with specific GPU index, e.g.
```bash
CUDA_IDX=0
```

## Simulation 1: Toy example
**Figure 2.** Comparison of  Comparison of Analog SGD and Tiki-Taka under different parameter $c_{\text{Lin}}$
```bash
python S1-TT-fails.py
```

## Simulation 2: FCN/CNN @ MNIST
**Figure 4.** 
The network archetecture can be fully-connected network (FCN) or convolutional neural network (CNN)

As pointed out in the paper, RL-v1 can be implemented by TT-v1.

Perform simulations on FCN
```bash
python S1-mnist-FCN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.5
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S1-mnist-FCN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7
```
Perform simulations on CNN
```bash
python S2-mnist-CNN.py --SETTING="FP SGD" --CUDA=${CUDA_IDX}
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="Analog SGD" --CUDA=${CUDA_IDX} --tau=0.8
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.6
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.7
python S2-mnist-CNN.py --SETTING="TT-v1" --CUDA=${CUDA_IDX} --tau=0.8
```

## Simulation 3: Resnet/MobileNet @ CIFAR
```bash
# Analog SGD
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet18" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet34" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet50" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet18" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet34" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet50" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
# RLv1/TTv1
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet18" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet34" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="Resnet50" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet18" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet34" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="Resnet50" -TM="FFT" --optimizer="TT-v1" -lr=0.15  --tau=0.1 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
```

```bash
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV2"  -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV3L" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV3S" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV2"  -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV3L" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV3S" -TM="FFT" --optimizer="Analog SGD" -lr=0.15 --tau=0.05 --CUDA=${CUDA_IDX}

python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV2"  -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV3L" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR10"  --model="MobileNetV3S" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV2"  -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV3L" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
python A12-resnet-finetune.py --dataset="CIFAR100" --model="MobileNetV3S" -TM="FFT" --optimizer="TT-v1" -lr=0.15 --tau=0.05 --TTv1-gamma=0.4 --CUDA=${CUDA_IDX}
```

## Simulation 4: Ablation of $gamma$
```bash
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.1 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.1 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.2 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.2 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.3 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.3 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR10"  --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.4 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
python S3-resnet.py --dataset="CIFAR100" --model="Resnet18" -FFT --optimizer="TT-v1" --TTv1-gamma=0.4 --RPU=Exp  --tau=0.1 --res-gamma=3. --CUDA=${CUDA_IDX}
```


# Note
The implementation of RLv2 and power/exponential response functions have not been organized well so far since the code need to modify the code in AIHWKIT. Feel free to send me an email if you need any suggestion for reproducing the results.