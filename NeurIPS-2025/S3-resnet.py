import random
import argparse
import time
import os
from enum import Enum
import copy

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import models

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import sys
# sys.path.insert(0, '../aihwkit/src/')
import aihwkit
from utils.logger import Logger
print('aihwkit path: ', aihwkit.__file__)
print('[Running]', ' '.join(sys.argv))

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
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import (
    VectorUnitCellUpdatePolicy,
    NoiseManagementType,
    BoundManagementType,
)

parser = argparse.ArgumentParser(description="A simple command-line argument example")
parser.add_argument('--dataset', type = str, default = 'CIFAR10', help = 'use which dataset')
parser.add_argument('-opt', '--optimizer', type=str, help="", default='SGD')
parser.add_argument('-lr', '--lr', type=float, help="", default=1.5e-2)
parser.add_argument('-CUDA', '--CUDA', type=int, help="", default=-1)
parser.add_argument('-save', '--save-checkpoint', action='store_true')
parser.add_argument('-TM', '--train-mode', type=str, help="", default="FFT")
parser.add_argument('-init', '--init', type=str, help="", default=None)
parser.add_argument('-RPU', '--RPU', type=str, help="", default='Softbounds')
parser.add_argument('-tau', '--tau', type=float, help="", default=4)
parser.add_argument('-res-gamma', '--res-gamma', type=float, help="", default=-1)
parser.add_argument('-res-state', '--res-state', type=float, help="", default=None)
parser.add_argument('-model', '--model', type=str, help="", default="Resnet18")
parser.add_argument('-TT-gamma', '--TTv1-gamma', type=float, help="", default=-1)
parser.add_argument('-TTv2-bm', '--TTv2-buffer-as-momentum', action='store_true')
parser.add_argument('-TTv2-AG', '--TTv2-auto-granularity', type=float, help="", default=0)
parser.add_argument('--io-perfect', '--io-perfect', action='store_true')
parser.add_argument('-io-perfect-F', '--io-perfect-forward', action='store_true')
parser.add_argument('-io-perfect-B', '--io-perfect-backward', action='store_true')
parser.add_argument('-io-perfect-T', '--io-perfect-transfer', action='store_true')
parser.add_argument('--io-inp-res-bit', type=int, help="", default=-1)
parser.add_argument('--io-out-res-bit', type=int, help="", default=-1)
parser.add_argument('--io-inp-noise', type=float, help="", default=-1)
parser.add_argument('--io-out-noise', type=float, help="", default=-1)
args = parser.parse_args()

args_dict = copy.deepcopy(args.__dict__)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f"cuda:{args.CUDA}" if USE_CUDA else "cpu")
print('Using Device: ', DEVICE)

class opt_T(Enum):
    TORCH = 1
    KIT_FP = 2
    KIT_ANALOG = 3
    
valid_traing_mode = [
    'FFT', # replace the last layer, and finetune that whole model
    'FIX_BACKBONE', # replace the last layer but only update the last layer
    'FIX_ANALOG', # only train the digital part
    'FULL_ANALOG', # place the whole model on analog 
    'FULL_ANALOG_FS', # place the whole model on analog and train from scratch 
]
train_mode = args.train_mode
assert train_mode in valid_traing_mode, f'unknown training mode: {train_mode}'

EPOCHS = 100
    
tau = args.tau
optimizer_str = args.optimizer
save_checkpoint = args.save_checkpoint
print('[Save checkpoint]', save_checkpoint)
RPU_NAME = args.RPU
MODEL_NAME = args.model
if args.io_inp_res_bit != -1:
    io_inp_res = 1/(2**args.io_inp_res_bit - 2)
if args.io_out_res_bit != -1:
    io_out_res = 1/(2**args.io_out_res_bit - 2)

def get_opt_type(optimizer_str):
    torch_list = [
        'SGD', 'SGD-plain', 'AdamW'
    ]
    fp_list = [
        'FP SGD', 'FP SGDM'
    ]
    analog_list = [
        'Analog SGD', 'TT-v1', 'TT-v2', 'TT-v3', 'TT-v4', 'mp'
    ]
    if optimizer_str in torch_list:
        return opt_T.TORCH
    elif optimizer_str in fp_list:
        return opt_T.KIT_FP
    elif optimizer_str in analog_list:
        return opt_T.KIT_ANALOG
    else:
        assert False, "unknown algorithm type"
opt_type = get_opt_type(args.optimizer)

# lr = 256 * 5e-5
# lr = 1.5e-1 # too large
# lr = 1.5e-2 # proper
lr = args.lr
# lr = 0.001
# if opt_type is opt_T.KIT_ANALOG:
#     lr /= 2
    
DEFAULT_NUMBER_OF_STATES = 1200
if args.res_state is not None:
    num_of_states = args.res_state
else:
    num_of_states = DEFAULT_NUMBER_OF_STATES
    
class Cutout(object):
    """Random erase the given PIL Image.
    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`
    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        s_l (float): min cut square ratio. Default value is 0.02
        s_h (float): max cut square ratio. Default value is 0.4
        r_1 (float): aspect ratio of cut square. Default value is 0.4
        r_2 (float): aspect ratio of cut square. Default value is 1/0.4
        v_l (int): low filling num. Default value is 0
        v_h (int): high filling num. Default value is 255
        pixel_level (bool): filling one number or not. Default value is False
    """

    def __init__(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.4, r_2=1 / 0.4,
                 v_l=0, v_h=255, pixel_level=False):
        self.p = p
        self.s_l = s_l
        self.s_h = s_h
        self.r_1 = r_1
        self.r_2 = r_2
        self.v_l = v_l
        self.v_h = v_h
        self.pixel_level = pixel_level

    @staticmethod
    def get_params(img, s_l, s_h, r_1, r_2):

        img_h, img_w = img.size
        img_c = len(img.getbands())
        s = np.random.uniform(s_l, s_h)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = np.random.uniform(r_1, r_2)
        s = s * img_h * img_w
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        left, top, h, w, ch = self.get_params(img, self.s_l, self.s_h, self.r_1, self.r_2)

        if self.pixel_level:
            c = np.random.randint(self.v_l, self.v_h, (h, w, ch), dtype='uint8')
        else:
            c = np.random.randint(self.v_l, self.v_h) * np.ones((h, w, ch), dtype='uint8')
        c = Image.fromarray(c)
        img.paste(c, (left, top, left + w, top + h))
        return img

# from PIL import Image, ImageEnhance, ImageOps
# https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
# https://github.com/kakaobrain/fast-autoaugment
class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

image_size = 224
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])
BATCH_SIZE = 128
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False,download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=False,download=True, transform=transform_test)
    num_classes = 100
else:
    raise ValueError(f"unknown dataset type: {args.dataset}")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=6,pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=6)


def get_RPU_device(device_name=None):
    dw_min = 2 * tau / num_of_states
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'Softbounds':
        # by defauly
        # dw_min = 0.001
        # w_max = 0.6 w_min = -0.6
        # state = 1200
        return SoftBoundsReferenceDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau, 
            w_max_dtod=0., w_min_dtod=0.,
            # construction_seed=10
        )
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
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=tau, w_min=-tau,
            w_max_dtod=0., w_min_dtod=0.,
        )
    # elif device_name == 'RRAM':
    #     return get_RRAM()
    elif device_name == 'HfO2':
        from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
        return ReRamArrayHfO2PresetDevice()
    elif device_name == 'OM':
        from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
        return ReRamArrayOMPresetDevice()
    elif device_name == 'PCM':
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PCMPresetDevice()
    else:
        raise NotImplemented

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label
class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def get_optimizer_cls(optimizer_str):
    if optimizer_str == 'SGD':
        optimizer_cls = lambda params, lr, *args, **kwargs: optim.SGD(params, lr, momentum=0.9, *args, **kwargs)
    elif optimizer_str == 'SGD-plain':
        optimizer_cls = optim.SGD
    elif optimizer_str == 'AdamW':
        optimizer_cls = lambda params, lr, *args, **kwargs: optim.AdamW(params, lr, amsgrad=True, *args, **kwargs)
    else:
        assert False, f"unknown optimizer: {optimizer_str}"
    return optimizer_cls
def get_AnalogSGD_optimizer_generator(*args, **kargs):
    def _generator(params, lr):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator
def get_config(config_name):
    if config_name == 'FP SGD':
        rpu_config = FloatingPointRPUConfig()
        
        # FPSGD
        config = {
            'name': 'FP SGD',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
    elif config_name == 'FP SGDM':
        # FP GDM
        config = {
            'name': 'FPSGDM',
            'rpu_config': FloatingPointRPUConfig(),
            'optimizer_cls': get_AnalogSGD_optimizer_generator(momentum=0.99),
            'grad_per_iter': 1,
            # 'batch_size': DATASET_SIZE,
            'linestyle': '--',
        }
    elif config_name == 'Analog SGD':
        # Analog SGD
        rpu_config = SingleRPUConfig(
            # device=get_device('SB')
            device=get_RPU_device(RPU_NAME)
        )
        config = {
            'name': 'Analog SGD',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
        # update onto A matrix needs to be increased somewhat
        # rpu_config.mapping.weight_scaling_omega = 0.6
    elif config_name == 'TT-v1':
        algorithm = 'ttv1'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        # rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.01
        rpu_config.device.n_reads_per_transfer = 1
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        config = {
            'name': f'TT-v1',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
            
        if rpu_config.device.n_reads_per_transfer > 1:
            config['name'] += f'-st={rpu_config.device.n_reads_per_transfer}'
            
        if args.TTv1_gamma > 0:
            rpu_config.device.gamma = args.TTv1_gamma
            config['name'] += f'-g={args.TTv1_gamma:.2f}'
        
    elif config_name == 'TT-v2':
        rpu_device = get_RPU_device(RPU_NAME)
        # rpu_config = build_config(algorithm, device=device_config_fit, construction_seed=123)
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
            # **kwargs,
        )
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.3
        
        if args.TTv2_buffer_as_momentum:
            rpu_config.device.buffer_as_momentum = True
            rpu_config.device.momentum = 0.9
            rpu_config.device.auto_granularity=0
            rpu_config.device.fast_lr = 1
            rpu_config.device.scale_fast_lr = True
            rpu_config.device.transfer_lr = 0.1
            rpu_config.device.scale_transfer_lr = False
        else:
            # original
            rpu_config.device.transfer_lr = 0.08
            rpu_config.device.fast_lr = 0.01
            rpu_config.device.scale_fast_lr = False
            rpu_config.device.transfer_lr = 1.
            rpu_config.device.scale_transfer_lr = True
            # rpu_config.device.auto_granularity=15000
            rpu_config.device.auto_granularity=1000
            # rpu_config.device.auto_granularity=1000
        
        # if args.TTv2_auto_granularity > 0:
        rpu_config.device.auto_granularity = args.TTv2_auto_granularity
            
        config = {
            'name': f'TT-v2',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
        if args.TTv2_buffer_as_momentum:
            config['name'] += f'-bm'
            
        if args.TTv1_gamma > 0:
            rpu_config.device.gamma = args.TTv1_gamma
            config['name'] += f'-g={args.TTv1_gamma:.2f}'
            
        config['name'] += f'-ag={args.TTv2_auto_granularity}'
            
    elif config_name == 'TT-v3':
        algorithm = 'ttv3'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True
        # rpu_config.device.momentum=0.8

        
        config = {
            'name': f'TT-v3',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
    elif config_name == 'TT-v4':
        algorithm = 'ttv4'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
        # update onto A matrix needs to be increased somewhat
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_columnwise = True
        rpu_config.mapping.weight_scaling_omega = 0.6
        rpu_config.device.fast_lr = 0.1
        rpu_config.device.auto_granularity = 1000
        rpu_config.device.in_chop_prob = 0.1
        rpu_config.device.out_chop_prob = 0.
        rpu_config.device.auto_scale = True
        rpu_config.device.tail_weightening = 10
        # rpu_config.device.momentum=0.8
        
        config = {
            'name': f'TT-v4',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
    elif config_name == 'mp':
        algorithm = 'mp'  # one of tiki-taka, ttv2, c-ttv2, mp, sgd, agad
        rpu_device = get_RPU_device(RPU_NAME)
        rpu_config = build_config(algorithm, device=rpu_device, construction_seed=123)
        
        config = {
            'name': f'mp',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(),
            'grad_per_iter': 1,
        }
    else:
        raise NotImplementedError

    if opt_type is opt_T.KIT_ANALOG:
        
        if args.io_perfect:
            # perfect IO
            rpu_config.forward.is_perfect = True
            rpu_config.backward.is_perfect = True
            rpu_config.device.transfer_forward.is_perfect = True
        else:
            # imperfect IO
            # for each of forward, backward, transfer, deal with input/output, resistively
            # ===================================================================
            # forward
            config_IO(rpu_config.forward, {
                "io_perfect":     args.io_perfect_forward,
                "io_inp_res_bit": args.io_inp_res_bit,
                "io_out_res_bit": args.io_out_res_bit,
                "io_inp_noise": args.io_inp_noise,
                "io_out_noise": args.io_out_noise,
            })
            # backward
            config_IO(rpu_config.backward, {
                "io_perfect":     args.io_perfect_backward,
                "io_inp_res_bit": args.io_inp_res_bit,
                "io_out_res_bit": args.io_out_res_bit,
                "io_inp_noise": args.io_inp_noise,
                "io_out_noise": args.io_out_noise,
            })
            # transfer
            if config_name.startswith('TT'):
                config_IO(rpu_config.device.transfer_forward, {
                    "io_perfect":     args.io_perfect_backward,
                    "io_inp_res_bit": args.io_inp_res_bit,
                    "io_out_res_bit": args.io_out_res_bit,
                    "io_inp_noise": args.io_inp_noise,
                    "io_out_noise": args.io_out_noise,
                })
    
    return config

    
def config_IO(io_param: "aihwkit.simulator.parameters.IOParameters", config):
    """Set the IO parameters for the config"""
    if config["io_perfect"]:
        io_param.is_perfect = True
    else:
        if config["io_inp_res_bit"] != -1:
            io_param.inp_res = config["io_inp_res_bit"]
        if config["io_out_res_bit"] != -1:
            io_param.out_res = config["io_out_res_bit"]
        if config["io_inp_noise"] != -1:
            io_param.inp_noise = config["io_inp_noise"]
        if config["io_out_noise"] != -1:
            io_param.out_noise = config["io_out_noise"]
    
def test_evaluation(model, val_set, criterion):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    predicted_ok = 0
    total_images = 0

    model.eval()

    total_loss = 0
    with torch.no_grad():
        
        for images, labels in val_set:
            # Predict image.
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # images = images.view(images.shape[0], -1)
            pred = model(images)

            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            loss = criterion(pred, labels)
            total_loss += loss.item()

    # print("\nNumber Of Images Tested = {}".format(total_images))
    # print("Model Accuracy = {}".format(predicted_ok / total_images))
    loss = total_loss / total_images
    accuracy = predicted_ok / total_images
    return loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
def train():
    if MODEL_NAME == 'Resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'Resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'Resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'Resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'Resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    # elif MODLE_NAME == 'MobileNetV1w1':
    #     model = models.mobilenet_w1(pretrained=True)
    # elif MODLE_NAME == 'MobileNetV1w3d4':
    #     model = models.mobilenet_wd2(pretrained=True)
    # elif MODLE_NAME == 'mobilenetV1wd2':
    #     model = models.mobilenet_wd2(pretrained=True)
    # elif MODLE_NAME == 'mobilenetV1wd4':
    #     model = models.mobilenet_wd4(pretrained=True)
    elif MODEL_NAME == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'MobileNetV3L':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif MODEL_NAME == 'MobileNetV3S':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown model: {MODEL_NAME}")
    if train_mode == 'FIX_BACKBONE':
        for param in model.parameters():
            param.requires_grad = False
            
    # Get the input size of the last layer
    if MODEL_NAME[:6] == 'Resnet':
        num_ftrs = model.fc.in_features
    elif MODEL_NAME[:9] == 'MobileNet':
        num_ftrs = model.classifier[-1].in_features
    else:
        raise ValueError(f"unknown model: {MODEL_NAME}")
    print(f'num_ftrs: {num_ftrs}')
    
    dataset_name = f'{args.dataset}-finetune/{MODEL_NAME}/{train_mode}'
        
    if opt_type is opt_T.TORCH:
        alg_name = f'torch-{args.optimizer}'
        config = None
        path_name = f'{dataset_name}/GPU'
        
        # replace the last layer
        replaced_layer = nn.Linear(num_ftrs, num_classes)
        optimizer_cls = get_optimizer_cls(args.optimizer)
        # optimizer = get_optimizer(model.fc, args.optimizer)
        
        assert train_mode != 'FULL_ANALOG'
    else:
        # if USE_CUDA:
        #     model.cuda(DEVICE)
        config = get_config(optimizer_str)
        rpu_config = config['rpu_config']
        
        # name of the record
        alg_name = config['name']
        if opt_type is opt_T.KIT_ANALOG:
            alg_name += f'-tau={tau}'
            RPU_suffix = args.RPU
            if args.RPU == 'Pow' and args.res_gamma > 0:
                RPU_suffix += f'-gamma={args.res_gamma:.2}'
            if args.RPU == 'Exp' and args.res_gamma > 0:
                RPU_suffix += f'-gamma={args.res_gamma:.2}'
            if args.res_state is not None:
                RPU_suffix += f'-state={int(args.res_state)}'
            path_name = os.path.join(dataset_name, RPU_suffix)
        elif opt_type is opt_T.KIT_FP:
            path_name = dataset_name
        else:
            raise ValueError(f"unknown opt_type: {opt_type}")
        
        # prefix for IO
        if args.io_perfect:
            alg_name += f'-ioperfect'
        else:
            if args.io_perfect_forward or args.io_perfect_backward or args.io_perfect_transfer:
                alg_name += f'-io'
                if args.io_perfect_forward:
                    alg_name += f'F'
                if args.io_perfect_backward:
                    alg_name += f'B'
                if args.io_perfect_transfer:
                    alg_name += f'T'
            else:
                if args.io_inp_res_bit != -1:
                    alg_name += f'-inpres={args.io_inp_res_bit}'
                if args.io_out_res_bit != -1:
                    alg_name += f'-outres={args.io_inp_res_bit}'
            if args.io_out_noise != -1:
                alg_name += f'-outnoise={args.io_out_noise}'
        
        # replace the last layer
        if train_mode == 'FULL_ANALOG':
            replaced_layer = nn.Linear(num_ftrs, num_classes)
        else:
            replaced_layer = AnalogLinear(num_ftrs, num_classes, rpu_config=rpu_config)
        # model.fc = AnalogSequential(
        #     AnalogLinear(num_ftrs, HIDDEN_DIM, rpu_config),
        #     nn.Tanh(),
        #     AnalogLinear(HIDDEN_DIM, num_classes, rpu_config),
        # )
        optimizer_cls = config['optimizer_cls']
    
    if args.lr != 1e-2:
        alg_name += f'-blr={args.lr}'
    if args.init is not None:
        alg_name += f'-init={args.init}'
    
    if MODEL_NAME[:6] == 'Resnet':
        model.fc = replaced_layer
    elif MODEL_NAME[:9] == 'MobileNet':
        model.classifier[-1] = replaced_layer
    else:
        raise ValueError(f"unknown model: {MODEL_NAME}")
    
    if train_mode == 'FULL_ANALOG':
        model = convert_to_analog(model, rpu_config=rpu_config)

    if USE_CUDA:
        model = model.to(DEVICE)
    # optimizer = optimizer_cls(model.fc.parameters(), lr)
    # params_1x = [param for name, param in model.named_parameters()
    #         if name.split('.')[0] != 'fc'
    # ]
    # optimizer = optimizer_cls([
    #         {'params': params_1x, 'lr': lr / 10},
    #         {'params': replaced_layer.parameters()}
    #     ], lr)
    
    if args.train_mode == 'FIX_ANALOG':
        for param in model.parameters():
            if isinstance(param, AnalogContext):
                in_size = param.analog_tile.in_size
                out_size = param.analog_tile.out_size
                # zero
                # init_weights = torch.zeros(out_size, in_size)
                
                # xavier
                init_weights = torch.empty(out_size, in_size)
                nn.init.xavier_uniform_(init_weights, gain=1)
                
                param.analog_tile.set_weights(init_weights)
                param.requires_grad = False
        params = [param for param in model.parameters() if not isinstance(param, AnalogContext)]
    else:
        params = list(model.parameters())
        
    if args.init is not None:
        for param in model.parameters():
            if not isinstance(param, AnalogContext):
                continue
            if args.init == 'xavier':
                in_size = param.analog_tile.in_size
                out_size = param.analog_tile.out_size
                init_weights = torch.empty(out_size, in_size)
                nn.init.xavier_uniform_(init_weights, gain=1)
            elif args.init == 'zero':
                in_size = param.analog_tile.in_size
                out_size = param.analog_tile.out_size
                init_weights = torch.zeros(out_size, in_size)
            elif args.init == 'normal':
                in_size = param.analog_tile.in_size
                out_size = param.analog_tile.out_size
                init_weights = torch.empty(out_size, in_size)
                nn.init.normal_(init_weights, mean=0.0, std=0.01)
            else:
                raise ValueError(f"unknown init method: {args.init}")
    
    # fail
    # num_fine_tune = len(list(replaced_layer.parameters()))
    # optimizer = optimizer_cls([
    #         {'params': params[:-num_fine_tune], 'lr': lr / 10},
    #         {'params': params[-num_fine_tune]}
    #     ], lr)
    # optimizer = optimizer_cls([
    #         {'params': params[:-1], 'lr': lr / 10},
    #         {'params': params[-1]}
    #     ], lr)
    optimizer = optimizer_cls(params, lr)
    # num_fine_tune = len(list(replaced_layer.parameters()))
    # optimizer = optimizer_cls([
    #         {'params': params[:-num_fine_tune], 'lr': lr / 10},
    #         {'params': params[-num_fine_tune:]}
    #     ], lr)
    
    log_path = f'runs/{path_name}/{alg_name}'
    logger = Logger(log_path)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(10)
    # criterion = LabelSmoothingLoss(10, 0.1)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    allstart = time.time()
    test_loss, test_accuracy = test_evaluation(model, testloader, criterion)
    log_str = f"Epoch {0} - Training loss: --------   Test Accuracy: {test_accuracy:.4f}"
    logger.write(0, log_str, {
        "Loss/test": test_loss,
        "Accuracy/test": test_accuracy
    })

    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Decay learning rate if needed.
        scheduler.step()
        
        train_loss = total_loss / len(trainloader)
        test_loss, test_accuracy = test_evaluation(model, testloader, criterion)
        
        log_str = f"Epoch {epoch} - Training loss: {train_loss:.6f}   Test Accuracy: {test_accuracy:.4f}"
        logger.write(epoch, log_str, {
            "Loss/train": train_loss,
            "Loss/test": test_loss,
            "Accuracy/test": test_accuracy,
            "State/lr": scheduler.get_last_lr()[0],
        })
          
    print(f'Finished Training: {path_name}/{alg_name}')
    allend = time.time()
    print("time: ", allend - allstart)
    if args.save_checkpoint:
        check_point_folder = f'checkpoints/{path_name}'
        checkpoint_path = f'{check_point_folder}/{alg_name}.pth'
        if not os.path.isdir(check_point_folder):
            os.makedirs(check_point_folder)
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        print(f'Save checkpoint: {checkpoint_path}')

    
if __name__ ==  '__main__':
    train()