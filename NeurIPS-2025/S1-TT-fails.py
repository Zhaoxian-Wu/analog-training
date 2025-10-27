# %%
import math

import torch
import torch.nn as nn
import torch.optim as optim
import os

from matplotlib.lines import Line2D

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext

from aihwkit.optim import AnalogSGD

from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    SingleRPUConfig,
    WeightNoiseType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.devices import (
    SoftBoundsReferenceDevice,
    TransferCompound,
)
from aihwkit.simulator.parameters import (
    IOParameters,
)
import argparse

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Parse the command line arguments
args = parser.parse_args()

# lr = 0.1 is a baseline lr
lr = 0.8
fast_lr = 1e-1 * (lr / 0.1)
gamma = 0.4
DESIRED_BL = 8
# DESIRED_BL = 100

gradient_budget = 1000
# gradient_budget = 300 * int(0.1 / lr)
if lr < 0.1:
    gradient_budget *= int(0.1 / lr)

INPUT_SIZE = 50
DATASET_SIZE = 100
torch.manual_seed(42)  # For reproducibility
element_scale = 0.5
groundtrue = element_scale * torch.randn(INPUT_SIZE, requires_grad=False)
bias_true = 0
print('[DIM]', groundtrue.numel())
print('average scale:', groundtrue.norm(p=1) / groundtrue.numel())
print('max scale:', groundtrue.max())
A = torch.randn((DATASET_SIZE, INPUT_SIZE), requires_grad=False)
Y = torch.matmul(A, groundtrue) + bias_true
print('max data scale:', A.max())

criterion = nn.MSELoss()

def get_weights(model):
    weights_list = []
    for param in model.parameters():
        if isinstance(param, AnalogContext):
            tile = param.analog_tile
            weight, bias = tile.get_weights()
            weights_list.append(weight)
        else:
            weights_list.append(param.data.detach())
    return weights_list
def init_model(model, init_value=1.):
    for param in model.parameters():
        if isinstance(param, AnalogContext):
            tile = param.analog_tile
            w_z = init_value * torch.ones(tile.in_size, tile.out_size)
            b_z = init_value * torch.ones(tile.out_size)
            tile.set_weights(w_z, b_z)
        else:
            torch.nn.init.constant_(param, init_value)
def get_device(w_min, w_max, up_down):
    device = SoftBoundsReferenceDevice(
        construction_seed=10,
        dw_min = 1e-4,
        dw_min_dtod = 0,
        dw_min_std = 0,
        dw_min_dtod_log_normal=False,
        write_noise_std = 0,
        corrupt_devices_prob = 0,
        corrupt_devices_range = 0,
        up_down = up_down,
        up_down_dtod = 0,
        slope_up_dtod = 0,
        slope_down_dtod = 0,
        reference_mean = 0,
        reference_std = 0,
        w_max = w_max,
        w_min = w_min,
        w_min_dtod = 0,
        w_max_dtod = 0,
        reset_std = 0,
        perfect_bias = False,
    )
    return device
def get_model(rpu_config=None):
    if rpu_config is None:
        model = torch.nn.Linear(INPUT_SIZE, 1, False)
    else:
        model = AnalogLinear(INPUT_SIZE, 1, False, rpu_config=rpu_config)
    return model
digital_temp_model = get_model()
def get_loss(model, for_training=True, analog_exact=True):
    if (not for_training) and analog_exact and isinstance(model, AnalogLinear):
        for dig_param, analog_param in zip(digital_temp_model.parameters(), model.parameters()):
            if isinstance(analog_param, AnalogContext):
                weights = analog_param.analog_tile.get_weights()[0]
            else:
                weights = analog_param.data
            dig_param.data.copy_(weights)
        model = digital_temp_model
    outputs = model(A)
    loss = criterion(outputs.view(-1), Y) / 2
    if for_training:
        return loss
    else:
        return loss / INPUT_SIZE
def get_IO(noise_std=0):
    io_param = IOParameters(
        is_perfect  = False,
        inp_bound  = 10,
        out_bound  = 10,
        w_noise    = noise_std,
        w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT,
        inp_noise  = 0.,
        out_noise  = 0.,
        inp_res    = 0,
        out_res    = 0,
        ir_drop    = 0,
        ir_drop_g_ratio=0,
        noise_management = NoiseManagementType.NONE,
        bound_management = BoundManagementType.NONE,
        v_offset_w_min = 0,
    )
    return io_param
    
def run_SGD(noise_std=0.1):
    model = get_model(None)
    init_model(model)

    optimizer = optim.SGD(model.parameters(), lr=lr)  # Learning rate

    loss_list = []
    weight_list = []
    # Training loop
    for epoch in range(gradient_budget):
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        weight = get_weights(model)
        weight_list.append(weight)
        
        
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.data 
            grad.add_(noise_std * torch.randn_like(grad))

        optimizer.step()
        optimizer.zero_grad()
    return loss_list, weight_list

def run_TT(w_min=-4, w_max=4, up_down_variation=0, noise_std=0.1, record_weight=False):
    units_in_mbatch = True
    construction_seed = 23
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            # devices that compose the Tiki-taka compound
            unit_cell_devices=[
                get_device(w_min=w_min, w_max=w_max, up_down=up_down_variation),
                get_device(w_min=w_min, w_max=w_max, up_down=up_down_variation),
            ],

            # Make some adjustments of the way Tiki-Taka is performed.
            units_in_mbatch=units_in_mbatch,   # batch_size=1 anyway
            transfer_every=1,       # every 2 batches do a transfer-read
            n_reads_per_transfer=INPUT_SIZE,  # one forward read for each transfer
            gamma=gamma,              # all SGD weight in second device
            scale_transfer_lr=True, # in relative terms to SGD LR
            transfer_lr=1,        # momentum coefficient
            fast_lr=fast_lr,
            construction_seed = construction_seed,
        )
    )
    
    rpu_config.forward  = get_IO(noise_std=noise_std)
    rpu_config.backward = get_IO(noise_std=noise_std)
    rpu_config.transfer_forward = get_IO(noise_std=noise_std)
    rpu_config.update.desired_bl = DESIRED_BL
    rpu_config.update.sto_round = True
    
    torch.manual_seed(618)
    # <<<<<<<< Define model <<<<<<<<
    model = get_model(rpu_config)
    init_model(model)
    # <<<<<<<< Define optimizer <<<<<<<<
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    loss_list = []
    weight_list = []
    for iter_idx in range(gradient_budget+1):
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)
        if record_weight:
            weight = get_weights(model)
            weight_list.append(weight)
        model.train()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    if record_weight:
        return loss_list, weight_list
    else:
        return loss_list

def run_AGD(w_min=-4, w_max=4, up_down_variation=0, noise_std=0.1, record_weight=False):
    rpu_config = SingleRPUConfig(
        device=get_device(w_min=w_min, w_max=w_max, up_down=up_down_variation)
    )
    
    rpu_config.forward = get_IO(noise_std)
    rpu_config.backward = get_IO(noise_std)
    rpu_config.update.desired_bl = DESIRED_BL
    rpu_config.update.sto_round = True
    
    torch.manual_seed(618)
    # <<<<<<<< Define model <<<<<<<<
    model = get_model(rpu_config)
    init_model(model)
    # <<<<<<<< Define optimizer <<<<<<<<
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    loss_list = []
    weight_list = []
    for iter_idx in range(gradient_budget+1):
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)
        if record_weight:
            weight = get_weights(model)
            weight_list.append(weight)
        model.train()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
    for analog_tile in model.analog_tiles():
        tile_weight = analog_tile.get_hidden_parameters()
    print(f'max_bound: {tile_weight["max_bound"].mean().item()}, min_bound: {tile_weight["min_bound"].mean().item()}')

    if record_weight:
        return loss_list, weight_list
    else:
        return loss_list


import matplotlib.pyplot as plt
def plot_loss_curves(loss_data, weight_data, legend_lines, legend_labels, linestyles, colors, markers):
    """
    Plot multiple loss curves on the same graph.

    Parameters:
    - loss_data: A list of lists/tuples, where each inner list/tuple contains loss values.
    - labels: A list of labels corresponding to each loss curve.
    """
    plt.figure(figsize=(4, 3))  # Adjust the figure size as needed
    
    for loss_idx, loss_values in enumerate(loss_data): 
        label = None
        linestyle = linestyles[loss_idx]
        color = colors[loss_idx]
        marker = markers[loss_idx]
        
        smooth_loss = []
        for loss in loss_values:
            if len(smooth_loss) == 0:
                smooth_loss.append(loss.detach().item())
            else:
                smooth_coef = 0.9
                res = smooth_coef*smooth_loss[-1]+(1-smooth_coef)*loss
                smooth_loss.append(res.detach().item())
        plt.plot(smooth_loss, 
                 label=label, linestyle=linestyle, color=color,
                 marker=marker, markevery=100)
    plt.xlabel("Number of Gradient Computation (k)")
    plt.ylabel("$f(W_k)$")
    plt.legend(
        legend_lines, legend_labels,
        bbox_to_anchor=(1,0), loc="lower left",
    )
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplots_adjust(hspace=0.6)
    pic_name = 'TT-fails'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'fig', 'png')
    dir_pdf_path = os.path.join(file_dir, 'fig', 'pdf')
    print(file_dir)

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)
    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')

    print(f'finish: {pic_name}')

tau = 3.5
noise_std = 0. / math.sqrt(INPUT_SIZE)

linestyles = []
colors = []
markers = []

kit_noise_std = noise_std
results = []

up_down_list = [0, 0.1, 0.2, 0.3]
    
legend_lines = [
    Line2D([0], [0], color=f'C{idx}', lw=4)
    for idx in range(len(up_down_list))
]
legend_lables = [
    r'$c_{\text{Lin}}=' + f'{up_down:.1f}$' for up_down in up_down_list
]

TT_LINESTYLE = '-'
TT_MARKER = '*'
ASGD_LINESTYLE = '--'
ASGD_MARKER = 'o'

legend_lines.append(
    Line2D([0], [0], linestyle=TT_LINESTYLE,
            marker=TT_MARKER,
            markersize=4,
            color=f'k')
)
legend_lables.append('Tiki-Taka')

legend_lines.append(
    Line2D([0], [0], linestyle=ASGD_LINESTYLE,
            marker=ASGD_MARKER,
            markersize=4,
            color=f'k')
)
legend_lables.append('Analog SGD')

for up_down_idx, up_down in enumerate(up_down_list):
    results.append(run_AGD(-tau, tau, up_down, kit_noise_std))
    linestyles.append('-')
    colors.append(f'C{up_down_idx}')
    markers.append('*')
for up_down_idx, up_down in enumerate(up_down_list):
    results.append(run_TT(-tau, tau, up_down, kit_noise_std))
    linestyles.append('--')
    colors.append(f'C{up_down_idx}')
    markers.append('o')

plot_loss_curves(results, None, legend_lines, legend_lables, linestyles, colors, markers)
