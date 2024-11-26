import torch
# pylint: disable=invalid-name

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import (
    build_config,
    UnitCellRPUConfig,
    DigitalRankUpdateRPUConfig,
    FloatingPointRPUConfig,
    SingleRPUConfig,
    UpdateParameters,
)
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice,
    ConstantStepDevice,
    VectorUnitCell,
    LinearStepDevice,
    SoftBoundsDevice,
    SoftBoundsReferenceDevice,
    TransferCompound,
    MixedPrecisionCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda:2" if USE_CUDA else "cpu")
print('Using Device: ', DEVICE)

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 80
BATCH_SIZE = 10

# DEVICE_NAME = 'PCM'
DEVICE_NAME = 'Softbounds'
# DEVICE_NAME = 'HfO2'
# DEVICE_NAME = 'RRAM-offset'
dataset_name = 'MNIST-FCN'
tau = 0.7

def get_device(device_name='CS'):
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        return SoftBoundsDevice(w_max=tau, w_min=-tau, construction_seed=10)
    elif device_name == 'LS':
        return LinearStepDevice(w_max_dtod=0.4)
    elif device_name == 'HfO2':
        from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
        return ReRamArrayHfO2PresetDevice()
    elif device_name == 'PCM':
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PCMPresetDevice()
    else:
        raise NotImplemented
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
        # nn.ReLU(),
        torch.nn.Linear(
            hidden_sizes[0],
            hidden_sizes[1],
            True,
        ),
        nn.Sigmoid(),
        # nn.ReLU(),
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
    optimizer = AnalogSGD(model.parameters(), lr=0.05)
    optimizer.regroup_param_groups(model)

    return optimizer
@torch.no_grad()
def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()
    classifier = nn.NLLLoss()

    total_loss = 0
    for images, labels in val_set:
        # Predict image.
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
def get_config(config_name):
    if config_name == 'FP SGD':
        # FP SGD
        config = {
            'name': 'FP SGD',
            'rpu_config': FloatingPointRPUConfig(),
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'FP SGDM':
        # FP GDM
        # Set the `batch_size` as full batch
        config = {
            'name': 'FP SGDM',
            'rpu_config': FloatingPointRPUConfig(),
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(momentum=0.99),
            'grad_per_iter': 1,
            # 'batch_size': DATASET_SIZE,
            'linestyle': '--',
        }
    elif config_name == 'Analog SGD':
        # Analog SGD
        rpu_config = SingleRPUConfig(
            device=get_device(DEVICE_NAME)
        )
        config = {
            'name': 'Analog SGD',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
    elif config_name == 'TT-v1':
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.01
        # rpu_config.device.auto_granularity=15000
        
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'TT-v2':
        algorithm = 'ttv2'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.auto_granularity=15000
        
        config = {
            'name': f'TT-v2',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'TT-v3':
        algorithm = 'ttv3'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.auto_granularity = 15000
        
        config = {
            'name': f'TT-v3',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'TT-v4':
        algorithm = 'ttv4'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.auto_granularity=15000
        
        config = {
            'name': f'TT-v4',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    elif config_name == 'mp':
        algorithm = 'mp'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        device_config_fit = get_device(DEVICE_NAME)
        rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
        
        config = {
            'name': f'mp',
            'rpu_config': rpu_config,
            # 'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
        }
    else:
        raise NotImplementedError
    # rpu_config = SingleRPUConfig(device=ConstantStepDevice())
    return config

# Load datasets.
train_dataset, validation_dataset = load_images()

def load_model(setting, check_point_path):
    config = get_config(setting)

    name = config['name']

    tensor_board_path = f'runs/{dataset_name}/{DEVICE_NAME}/{name}'

    print(tensor_board_path)
    rpu_config = config['rpu_config']
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, 
                                rpu_config=rpu_config)
    checkpoint = torch.load(check_point_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model
    

def get_statistic(model):
    weights_buffer = []
    for layer in model.analog_layers():
        weights, bias = layer.get_weights()
        weights_buffer.extend(weights.cpu().numpy().flatten())
            
    return model, weights_buffer

def get_saturation(model, device_idx=None):
    saturation_list = []
    for analog_tile in model.analog_tiles():
        hidden_param = analog_tile.get_hidden_parameters()
        if device_idx is None:
            max_key = f'max_bound'
            min_key = f'min_bound'
            weight = analog_tile.get_weights(apply_weight_scaling=False)[0]
        else:
            max_key = f'max_bound_{device_idx}'
            min_key = f'min_bound_{device_idx}'
            weight_key = f'hidden_weights_{device_idx}'
            weight = hidden_param[weight_key]
        max_bound = hidden_param[max_key]
        min_bound = hidden_param[min_key]
        saturation_tensor = torch.zeros_like(weight)
        saturation_tensor += (weight >= 0) * weight.abs() / max_bound.abs()
        saturation_tensor += (weight < 0) * weight.abs() / min_bound.abs()
        saturation_list.append(saturation_tensor.view(-1))
        if saturation_tensor.max() > 1:
            raise ValueError(f'Some tile saturation is incorrect (saturation_tensor.max()={saturation_tensor.max()})')
    saturation_vec = torch.cat(saturation_list)
    
    # avg_sat = saturation_vec.mean().item()
    # max_sat = saturation_vec.max().item()
    # print(f'[saturation (kit)] avg: {avg_sat:.4f} max: {max_sat:.4f}')
    return saturation_vec

plot_configs = []

# >>>>> print FP >>>>>
setting = 'FP SGD'
path_name = f'{dataset_name}/{DEVICE_NAME}'
check_point_folder = f'checkpoints/{path_name}'  
check_point_path = f'{check_point_folder}/{setting}.pth'
plot_configs.append({
    'setting': setting,
    'check_point_path': check_point_path,
    'label': 'Digital SGD',
})

plot_configs.append({
    'setting': 'Analog SGD',
    'check_point_path': f'checkpoints/MNIST-FCN/{DEVICE_NAME}/Analog SGD-tau={tau}.pth',
    'label': 'Analog GD',
})

plot_configs.append({
    'setting': 'TT-v1',
    'check_point_path': f'checkpoints/MNIST-FCN/{DEVICE_NAME}/TT-v1-tau={tau}.pth',
    'label': 'Tiki-Taka',
})

import matplotlib.pyplot as plt

def plot_multi(plot_config):
    SCALE = 1
    plt.figure(figsize=(SCALE*3.7, SCALE*3))
    
    for plot_config in plot_configs:
        setting = plot_config['setting']
        check_point_path = plot_config['check_point_path']
        label = plot_config['label']
        model = load_model(setting, check_point_path)
        model, weights = get_statistic(model)
        print(f'setting={setting}, # of weights: {len(weights)}')
        plt.hist(weights, bins=400, alpha=0.2, label=label,
                 density=True)

    # plt.axvline(x=tau, color='k', linestyle='--', label=rf'$\tau$')
    # plt.axvline(x=-tau, color='k', linestyle='--')
    plt.legend()
    plt.xlim(left=-1., right=1.)
    plt.xlabel('Weight Values')
    plt.ylabel('Frequently')
    file_dir = ''
    dir_png_path = os.path.join(file_dir, 'fig-dist', 'png')
    dir_pdf_path = os.path.join(file_dir, 'fig-dist', 'pdf')

    pic_name = f'A03-distribution-comp'
    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)
    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    print(f'finish: {pic_name}')
    plt.clf()

def plot_saturation_dist(plot_config):
    SCALE = 1
    plt.figure(figsize=(SCALE*3.7, SCALE*3))
    
    for plot_config in plot_configs:
        setting = plot_config['setting']
        check_point_path = plot_config['check_point_path']
        label = plot_config['label']
        model = load_model(setting, check_point_path)
        if setting == 'FP SGD':
            pass
        elif setting == 'Analog SGD':
            weights = get_saturation(model)
            print(f'setting={setting}, # of weights: {len(weights)}')
            plt.hist(weights, bins=400, alpha=0.2, label=label,
                    density=True)
        elif setting == 'TT-v1':
            weights = get_saturation(model, device_idx=1)
            # P = get_saturation(model, device_idx=0)
            print(f'setting={setting}, # of weights: {len(weights)}')
            plt.hist(weights, bins=400, alpha=0.2, label=label,
                    density=True)
            # plt.hist(P, bins=400, alpha=0.2, label=label+' ($P_k$)',
            #         density=True)
            
    # plt.axvline(x=tau, color='k', linestyle='--', label=rf'$\tau$')
    # plt.axvline(x=-tau, color='k', linestyle='--')
    # plt.title(f'Distribution of Linear Layer Weights')
    plt.legend()
    plt.xlim(left=0., right=1.)
    # plt.yscale('log')
    plt.xlabel('Saturation')
    plt.ylabel('Frequently')
    file_dir = ''
    dir_png_path = os.path.join(file_dir, 'fig-dist', 'png')
    dir_pdf_path = os.path.join(file_dir, 'fig-dist', 'pdf')

    pic_name = f'A03-saturation'
    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)
    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    print(f'finish: {pic_name}')
    # plt.show()
    plt.clf()

plot_saturation_dist(plot_configs)
plot_multi(plot_configs)
