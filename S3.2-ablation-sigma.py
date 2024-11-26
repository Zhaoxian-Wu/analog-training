# %%
import math

import torch
import torch.nn as nn
import torch.optim as optim
import os


from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext

from aihwkit.optim import AnalogSGD

from aihwkit.simulator.configs import (
    FloatingPointRPUConfig,
    SingleRPUConfig,
    WeightNoiseType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.devices import (
    SoftBoundsReferenceDevice,
)
from aihwkit.simulator.parameters import (
    IOParameters,
)
import argparse

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-BS', '--batch-size', type=int, help="", default=100)
parser.add_argument('--opt-gap', help="", action="store_true",
                    default=False)

# Parse the command line arguments
args = parser.parse_args()

# Create toy dataset
INPUT_SIZE = 50
DATASET_SIZE = 100
torch.manual_seed(42)  # For reproducibility

# ALG_STR = args.algorithm
BATCH_SIZE = args.batch_size
# DEVICE_NAME = args.device
assert BATCH_SIZE <= DATASET_SIZE

element_scale = 0.5
# element_scale = 4
x_true = element_scale * torch.randn(INPUT_SIZE, requires_grad=False)
# bias_true = element_scale * torch.randn(1, requires_grad=False)  # Noise term
bias_true = 0

print(x_true.numel())
print(x_true.norm(p=1) / x_true.numel())
A = torch.randn((DATASET_SIZE, INPUT_SIZE), requires_grad=False)
# A.div_(A.max() * 1.1)
Y = torch.matmul(A, x_true) + bias_true
print(A.max())

# Instantiate the model
# num_epochs = 5000
# gradient_budget = 2000
# gradient_budget = 800
gradient_budget = 200
criterion = nn.MSELoss()

lr = 0.1
# lr = 0.01

X_duplicated = torch.cat([A, A], dim=0)
Y_duplicated = torch.cat([Y, Y], dim=0)

def noise_correction(noise_std):
    B = torch.matmul(A.transpose(0, 1), A)
    # A_scale = (torch.linalg.svdvals(B)[0])
    A_scale = ((B**2).sum() / INPUT_SIZE).sqrt().item()
    return noise_std / A_scale
def init_model(model):
    for p in model.parameters():
        nn.init.constant_(p, 0)
def get_device(tau):
    device = SoftBoundsReferenceDevice(
        construction_seed=10,
        dw_min = 1e-4,
        # dw_min = 5e-5,
        dw_min_dtod = 0,
        dw_min_std = 0,
        dw_min_dtod_log_normal=False,
        write_noise_std = 0,
        corrupt_devices_prob = 0,
        corrupt_devices_range = 0,
        up_down = 0,
        up_down_dtod = 0,
        slope_up_dtod = 0,
        slope_down_dtod = 0,
        reference_mean = 0,
        reference_std = 0,
        w_max = tau,
        w_min = -tau,
        w_min_dtod = 0,
        w_max_dtod = 0,
        reset_std = 0,
        subtract_symmetry_point = True,
        perfect_bias = False,
    )
    return device
def get_model(rpu_config=None):
    if rpu_config is None:
        model = torch.nn.Linear(INPUT_SIZE, 1)
    else:
        model = AnalogLinear(INPUT_SIZE, 1, rpu_config=rpu_config)
    return model
digital_temp_model = get_model()
def get_loss(model, for_training=True, analog_exact=True):
    if (not for_training) and analog_exact and isinstance(model, AnalogLinear):
        for dig_param, analog_param in zip(digital_temp_model.parameters(), model.parameters()):
            # analog_param.analog_tile.set_weights(dig_param.data)
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
        # inp_noise_std = 0.,
        # out_noise_std = 0,
        # out_res    = 1/(2**15-2),
        inp_res    = 0,
        out_res    = 0,
        ir_drop    = 0,
        ir_drop_g_ratio=0,
        noise_management = NoiseManagementType.NONE,
        bound_management = BoundManagementType.NONE,
        v_offset_w_min = 0,
    )
    return io_param
def get_opt():
    running_config.append(FloatingPointRPUConfig())
    
    model = get_model(rpu_config)
    init_model(model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = AnalogSGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(2*gradient_budget):

        outputs = model(A)
        loss = criterion(outputs.view(-1), Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return loss.item()
def get_GD():
    model = get_model(rpu_config)
    init_model(model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)  # Learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Learning rate

    losses = []
    # Training loop
    for epoch in range(gradient_budget):

        outputs = model(A)
        loss = criterion(outputs.view(-1), Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().numpy())
    return losses
def get_batch_loss(model, beg, batch_size=BATCH_SIZE):
    outputs = model(X_duplicated[beg:beg+batch_size])
    loss = criterion(outputs.view(-1), Y_duplicated[beg:beg+batch_size])
    return loss
def get_closure_loss_with_batch_idx(model, beg, batch_size=BATCH_SIZE):
    if beg is None:
        return None
    else:
        return lambda : get_batch_loss(model, beg, batch_size)
def get_AnalogSGD_optimizer_generator(lr=lr, *args, **kargs):
    def _generator(params):
        return AnalogSGD(params, lr=lr, *args, **kargs)
    return _generator

noise_std = 0.2 / math.sqrt(INPUT_SIZE)
def get_running_config():
    units_in_mbatch = True
    construction_seed = 23

    running_config = []
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Tiki-taka
    # tau_list = [2, 3, 10, 20, 50]
    std_list = [1e-3, 1e-2, 1e-1, 1e0]
    for noise_idx, noise_var in enumerate(std_list):
        noise_std = math.sqrt(noise_var)
        rpu_config = SingleRPUConfig(device=get_device(tau=2))
        config = {
            # 'name': rf'$\tau={tau}$-SGD',
            'name': rf'$\sigma^2={noise_var}$',
            'rpu_config': rpu_config,
            'optimizer_cls': get_AnalogSGD_optimizer_generator(lr=lr),
            'grad_per_iter': 1,
            'batch_size': BATCH_SIZE,
            'plt_fig': {
                'linestyle': '-',
                'marker': '*',
                'color': f'C{noise_idx}',
            }
        }
        running_config.append(config)
        
        kit_noise_std = noise_std
        # kit_noise_std = noise_correction(noise_std)
        # rpu_config.forward = get_IO(0)
        rpu_config.forward = get_IO(kit_noise_std)
        # rpu_config.backward = get_IO(0)
        rpu_config.backward = get_IO(kit_noise_std)
        rpu_config.update.desired_bl = 5000
        rpu_config.update.sto_round = True
        
    return running_config

def run_SGD(noise_std=0.1):
    model = get_model(None)
    init_model(model)
    # print([p for p in model.parameters()])

    optimizer = optim.SGD(model.parameters(), lr=lr)  # Learning rate

    loss_list = []
    # Training loop
    for epoch in range(gradient_budget):
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.data 
            grad.add_(noise_std * torch.randn_like(grad))

        optimizer.step()
        optimizer.zero_grad()
    loss_recorded = get_loss(model, for_training=False)
    loss_list.append(loss_recorded)
    return loss_list

running_config = get_running_config()
loss_list = [[] for _ in running_config]
# opt = get_GD()[-1]
for ii, config in enumerate(running_config):
    alg_name = config['name']
    rpu_config = config['rpu_config']
    optimizer_cls = config['optimizer_cls']
    grad_per_iter = config['grad_per_iter']
    batch_size = config['batch_size']

    torch.manual_seed(618)

    # <<<<<<<< Define model <<<<<<<<
    # analog_device = get_device(DEVICE_NAME)
    model = get_model(rpu_config)
    init_model(model)


    # <<<<<<<< Define optimizer <<<<<<<<
    optimizer = optimizer_cls(model.parameters())
    if hasattr(optimizer, 'regroup_param_groups'):
        optimizer.regroup_param_groups(model)

    # Training loop
    batch_idx_list = torch.randint(0, DATASET_SIZE, 
                                   size=[gradient_budget+grad_per_iter+1])
    

    num_iteration = gradient_budget // grad_per_iter
    for iter_idx in range(num_iteration+1):
        # evaluate
        # l = loss.detach().numpy()
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list[ii].append(loss_recorded)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Forward pass
        if alg_name == 'STORM':
            next_batch_idx = batch_idx_list[iter_idx+1]
            closure = get_closure_loss_with_batch_idx(model, next_batch_idx,
                                                      batch_size=batch_size)
        else:
            closure = None
        optimizer.step(closure)
        optimizer.zero_grad()

    # print([param for param in model.parameters()])
    # if (epoch + 1) % 100 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

import matplotlib.pyplot as plt

def plot_loss_curves(loss_data, labels, plt_figs, grad_per_iterations):
    """
    Plot multiple loss curves on the same graph.

    Parameters:
    - loss_data: A list of lists/tuples, where each inner list/tuple contains loss values.
    - labels: A list of labels corresponding to each loss curve.
    """
    SCALE = 0.7
    plt.figure(figsize=(SCALE*4, SCALE*4))  # Adjust the figure size as needed
    
    show_opt_gap = args.opt_gap
    if show_opt_gap:
        opt_value = get_opt()

    for loss_values, label, plt_fig, grad_per_iteration in zip(loss_data, labels, plt_figs, grad_per_iterations):
        if show_opt_gap:
            loss_values = [loss-opt_value for loss in loss_values]
        
        num_grad = [grad_per_iteration*i for i, _ in enumerate(loss_values)]

        smooth_loss = []
        for loss in loss_values:
            loss = loss.detach().item()
            if len(smooth_loss) == 0:
                smooth_loss.append(loss)
            else:
                gamma = 0.8
                smooth_loss.append(gamma*smooth_loss[-1]+(1-gamma)*loss)
        plt.plot(num_grad, smooth_loss, label=label, markevery=25, **plt_fig)

    # plt.ylim(top=2, bottom=0)
    # plt.title("Loss Curves")
    # plt.xlabel("Number of Gradient Computation")
    plt.xlabel(r"Number of Gradient Computation $(k)$")
    plt.ylabel(r"$f(W_k)-f^*$", fontsize=10)
    plt.legend(
        # fontsize=18,
        bbox_to_anchor=(1,0), loc="lower left",
    )
    plt.yscale('log')
    plt.grid(True)

    pic_name = f'Analog_GD_diff-std'
    file_dir = ''
    dir_png_path = os.path.join(file_dir, 'fig', 'png')
    dir_pdf_path = os.path.join(file_dir, 'fig', 'pdf')

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

labels = [config['name'] for config in running_config]

linestyles = [{} if 'plt_fig' not in config.keys() else config['plt_fig'] 
              for config in running_config]
grad_per_iter = [
   config['grad_per_iter'] for config in running_config
] + [1]

# labels.append('SGD')
# loss_list.append(run_SGD(noise_std))
# linestyles.append({
#     'linestyle': '--',
#     # 'marker': '*',
#     'color': f'C{len(linestyles)}',
# })

plot_loss_curves(loss_list, labels, linestyles, grad_per_iter)
