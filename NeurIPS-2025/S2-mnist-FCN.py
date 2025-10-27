# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

"""

import os
from time import time
from enum import Enum

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import sys
# sys.path.insert(0, '../aihwkit/src/')
import aihwkit
print('aihwkit path: ', aihwkit.__file__)

from utils.logger import Logger

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import (
    build_config,
    UnitCellRPUConfig,
    FloatingPointRPUConfig,
    SingleRPUConfig,
)
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    LinearStepDevice,
    SoftBoundsReferenceDevice,
    TransferCompound,
    MixedPrecisionCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
)

import argparse

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-SETTING', '--SETTING', type=str, help="", default='FP SGD')
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-RPU', '--RPU', type=str, help="", default='Softbounds')
parser.add_argument('-SB-d2d', '--SB-d2d', type=float, help="", default=-1)
parser.add_argument('-SB-dw_min_std', '--SB-dw_min_std', type=float, help="", default=-1)
parser.add_argument('-tau', '--tau', type=float, help="", default=1)
parser.add_argument('-res-state', '--res-state', type=float, help="", default=None)
parser.add_argument('-res-gamma', '--res-gamma', type=float, help="", default=-1)
parser.add_argument('-TT-gamma', '--TTv1-gamma', type=float, help="", default=-1)
parser.add_argument('-save', '--save-checkpoint', action='store_true')
args = parser.parse_args()
setting = args.SETTING

# Check device
USE_CUDA = 0
if cuda.is_compiled() and args.CUDA >= 0:
    USE_CUDA = 1
DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
print(' '.join(sys.argv))
print('Using Device: ', DEVICE)

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 10

tau = args.tau
RPU_NAME = args.RPU


class opt_T(Enum):
    TORCH = 1
    KIT_FP = 2
    KIT_ANALOG = 3
def get_opt_type(optimizer_str):
    torch_list = [
        'SGD', 'SGD-plain', 'AdamW'
    ]
    fp_list = [
        'FP SGD', 'FP SGDM'
    ]
    analog_list = [
        'Analog SGD', 'TT-v1', 'TT-v2', 'TT-v3', 'TT-v4', 'mp', 'RL-v2',
    ]
    if optimizer_str in torch_list:
        return opt_T.TORCH
    elif optimizer_str in fp_list:
        return opt_T.KIT_FP
    elif optimizer_str in analog_list:
        return opt_T.KIT_ANALOG
    else:
        assert False, "unknown algorithm type"
opt_type = get_opt_type(setting)

lr = 0.1
if opt_type is opt_T.KIT_ANALOG:
    lr /= 2

DEFAULT_NUMBER_OF_STATES = 1200
if args.res_state is not None:
    num_of_states = args.res_state
else:
    num_of_states = DEFAULT_NUMBER_OF_STATES

def get_RPU_device(device_name):
    # by defauly
    # dw_min = 0.001, tau=0.6, state = 1200
    dw_min = 2 * tau / num_of_states
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        device = SoftBoundsReferenceDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau, 
        )
        if args.SB_dw_min_std > 0:
            device.dw_min_std = args.SB_dw_min_std
        if args.SB_d2d > 0:
            device.dw_min_dtod = args.SB_d2d
            device.w_max_dtod = args.SB_d2d
            device.w_min_dtod = args.SB_d2d
        return device
    elif device_name == 'Exp':
        from aihwkit.simulator.configs.devices import ExpStepDevice
        device = ExpStepDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau,
            w_max_dtod=0, w_min_dtod=0
        )
        if args.res_gamma > 0:
            device.gamma_up = args.res_gamma
            device.gamma_down = args.res_gamma
        return device
    elif device_name == 'Pow':
        from aihwkit.simulator.configs.devices import PowStepDevice
        device = PowStepDevice(
            dw_min=dw_min,
            pow_gamma_dtod=0,
            w_max=tau, w_min=-tau,
            w_max_dtod=0, w_min_dtod=0
        )
        if args.res_gamma > 0:
            device.pow_gamma = args.res_gamma
        return device
    elif device_name == 'LS':
        return LinearStepDevice(w_max_dtod=0.4)
    elif device_name == 'ReRamSB':
        from aihwkit.simulator.presets.devices import ReRamSBPresetDevice
        return ReRamSBPresetDevice()
    elif device_name == 'ReRamES':
        from aihwkit.simulator.presets.devices import ReRamESPresetDevice
        return ReRamESPresetDevice()
    elif device_name == 'EcRam':
        from aihwkit.simulator.presets.devices import EcRamPresetDevice
        return EcRamPresetDevice()
    elif device_name == 'EcRamMO':
        from aihwkit.simulator.presets.devices import EcRamMOPresetDevice
        return EcRamMOPresetDevice()
    elif device_name == 'HfO2':
        from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
        return ReRamArrayHfO2PresetDevice()
    elif device_name == 'OM':
        from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
        return ReRamArrayOMPresetDevice()
    elif device_name == 'PCM':
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PCMPresetDevice()
    elif device_name == 'test':
        from aihwkit.simulator.configs.devices import PowStepDevice
        dw_min = 0.001 * 2**args.res_gamma
        device = PowStepDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau
        )
        return device
    else:
        raise NotImplementedError
def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data
def create_analog_network(input_size, hidden_sizes, output_size, rpu_config):
    """Create the neural network using analog and digital layers.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(
            input_size,
            hidden_sizes[0],
            True,
        ),
        nn.Sigmoid(),
        torch.nn.Linear(
            hidden_sizes[0],
            hidden_sizes[1],
            True,
        ),
        nn.Sigmoid(),
        torch.nn.Linear(
            hidden_sizes[1],
            output_size,
            True,
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda(DEVICE)
    model = convert_to_analog(model, rpu_config)

    return model
def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
def train(model, train_set, config, logger, checkpoint_path):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    optimizer_cls = config['optimizer_cls']
    
    classifier = nn.NLLLoss()
    optimizer = optimizer_cls(model.parameters())
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    time_init = time()
    test_loss, test_accuracy = test_evaluation(model, validation_dataset)

    log_str = f"Epoch {0} - Training loss: --------   Test Accuracy: {test_accuracy:.4f}"
    logger.write(0, log_str, {
        "Loss/test": test_loss,
        "Accuracy/test": test_accuracy
    })
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()
        
        # Decay learning rate if needed.
        scheduler.step()

        train_loss = total_loss / len(train_set)
        test_loss, test_accuracy = test_evaluation(model, validation_dataset)
        
        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
        })

    if args.save_checkpoint:
        if not os.path.isdir(check_point_folder):
            os.makedirs(check_point_folder)
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
    print("\nTraining Time (s) = {}".format(time() - time_init))
@torch.no_grad()
def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    predicted_ok = 0
    total_images = 0

    model.eval()
    classifier = nn.NLLLoss()

    total_loss = 0
    for images, labels in val_set:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        loss = classifier(pred, labels)
        total_loss += loss.item()

    loss = total_loss / total_images
    accuracy = predicted_ok / total_images
    return loss, accuracy
def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
    def _generator(params):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator
construction_seed = 23
def get_config(config_name):
    if config_name == 'FP SGD':
        # FPSGD
        config = {
            'name': 'FP SGD',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'FP SGDM':
        # FP GDM
        # Set the `batch_size` as full batch
        config = {
            'name': 'FPSGDM',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(momentum=0.99),
            'grad_per_iter': 1,
            # 'batch_size': DATASET_SIZE,
            'linestyle': '--',
        }
    elif config_name == 'Analog SGD':
        rpu_config = SingleRPUConfig(
            device=get_RPU_device(RPU_NAME)
        )
        config = {
            'name': 'Analog SGD',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
    elif config_name == 'TT-v1':
        algorithm = 'ttv1'
        device_config_fit = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.n_reads_per_transfer = 1
        
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
            
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
            
        if args.TTv1_gamma > 0:
            rpu_config.device.gamma = args.TTv1_gamma
            config['name'] += f'-g={args.TTv1_gamma:.2f}'
            
    elif config_name == 'TT-v2':
        rpu_config = UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                in_chop_prob=0.0,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=123,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.scale_fast_lr = False
        rpu_config.device.transfer_lr = 1
        rpu_config.device.scale_transfer_lr = True
        rpu_config.device.auto_granularity = 1000
        
        config = {
            'name': f'TT-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        
    elif config_name == 'RL-v2':
        rpu_config = UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[get_RPU_device(RPU_NAME), get_RPU_device(RPU_NAME)],
                transfer_forward=IOParameters(
                    noise_management=NoiseManagementType.NONE,
                    bound_management=BoundManagementType.NONE,
                ),
                transfer_update=UpdateParameters(
                    desired_bl=1, update_bl_management=False, update_management=False
                ),
                in_chop_prob=0.0,
                units_in_mbatch=True,
                auto_scale=False,
                construction_seed=123,
            ),
            forward=IOParameters(),
            backward=IOParameters(),
            update=UpdateParameters(desired_bl=5),
        )
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        
        rpu_config.device.buffer_as_momentum = True
        rpu_config.device.momentum = 0.9
        rpu_config.device.fast_lr = 1
        rpu_config.device.scale_fast_lr = True
        rpu_config.device.transfer_lr = 0.11 + 1 / num_of_states
        rpu_config.device.scale_transfer_lr = False
        rpu_config.device.auto_granularity = 100
        
        config = {
            'name': f'RL-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        
        if args.TTv1_gamma > 0:
            rpu_config.device.gamma = args.TTv1_gamma + 1/num_of_states
            config['name'] += f'-g={args.TTv1_gamma:.2f}'
    
    elif config_name == 'TT-v3':
        algorithm = 'ttv3'
        device_config_fit = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True

        config = {
            'name': f'TT-v3',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'TT-v4':
        algorithm = 'ttv4'
        device_config_fit = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True
        rpu_config.device.tail_weightening = 10
        
        config = {
            'name': f'TT-v4',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'mp':
        algorithm = 'mp'
        device_config_fit = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        
        config = {
            'name': f'mp',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    else:
        raise NotImplementedError
    return config


config = get_config(setting)

dataset_name = 'MNIST-FCN'
alg_name = config['name']
if opt_type is opt_T.KIT_ANALOG:
    alg_name += f'-tau={tau}'
    if args.SB_dw_min_std > 0:
        alg_name += f'-c2c={args.SB_dw_min_std}'
    if args.SB_d2d > 0:
        alg_name += f'-d2d={args.SB_d2d}'
        
    RPU_suffix = args.RPU
    if args.RPU == 'Pow' and args.res_gamma > 0:
        RPU_suffix += f'-gamma={args.res_gamma:.2}'
    if args.RPU == 'Exp' and args.res_gamma > 0:
        RPU_suffix += f'-gamma={args.res_gamma:.2}'
    if args.res_state is not None:
        RPU_suffix += f'-state={int(args.res_state)}'
    path_name = os.path.join(dataset_name, RPU_suffix)
else:
    path_name = f'{dataset_name}'

rpu_config = config['rpu_config']
check_point_folder = f'checkpoints/{path_name}'
check_point_path = f'{check_point_folder}/{alg_name}.pth'
log_path = f'runs/{path_name}/{alg_name}'
logger = Logger(log_path)

# Load datasets.
train_dataset, validation_dataset = load_images()

# Prepare the model.
model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, 
                              rpu_config=rpu_config)

# Train the model.
train(model, train_dataset, config, logger, check_point_path)