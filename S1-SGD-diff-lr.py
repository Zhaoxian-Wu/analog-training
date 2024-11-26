# %%
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext
from aihwkit.optim import AnalogSGD

from aihwkit.simulator.configs import (
    SingleRPUConfig,
    WeightNoiseType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters import (
    IOParameters,
)
from aihwkit.simulator.configs.devices import (
    SoftBoundsReferenceDevice
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
INPUT_SIZE = 40
DATASET_SIZE = 100
torch.manual_seed(42)  # For reproducibility

BATCH_SIZE = args.batch_size
assert BATCH_SIZE <= DATASET_SIZE

element_scale = 0.45
x_true = element_scale * torch.randn(INPUT_SIZE, requires_grad=False)
# bias_true = element_scale * torch.randn(1, requires_grad=False)  # Noise term
bias_true = 0

print(x_true.numel())
print(x_true.norm(p=1) / x_true.numel())

A = torch.randn((DATASET_SIZE, INPUT_SIZE), requires_grad=False)
Y = torch.matmul(A, x_true) + bias_true

gradient_budget = 2500

X_duplicated = torch.cat([A, A], dim=0)
Y_duplicated = torch.cat([Y, Y], dim=0)
criterion = nn.MSELoss()

# Instantiate the model
def init_model(model, init_value=0):
    for param in model.parameters():
        if isinstance(param, AnalogContext):
            tile = param.analog_tile
            w_z = init_value * torch.ones(tile.in_size, tile.out_size)
            b_z = init_value * torch.ones(tile.out_size)
            tile.set_weights(w_z, b_z)
        else:
            torch.nn.init.constant_(param, init_value)
            
def analog_increment(tensor, increment, tau, weight_decay=1):
    with torch.no_grad():
        tensor.data.mul_(weight_decay).add_(increment - 1/tau*torch.abs(increment)*tensor)
    return tensor

class AnalogSGD_approx(torch.optim.Optimizer):
    def __init__(self,params,lr=0.1,
                 noise_std=0.1,
                 active_radius=3, sqrgradnorm={}):
        defaults = dict(lr=lr, active_radius=active_radius,
                        noise_std=noise_std,
                        sqrgradnorm=sqrgradnorm)
        super(AnalogSGD_approx, self).__init__(params, defaults)

    # Returns the state of the optimizer as a dictionary containing state and param_groups as keys
    def __setstate__(self,state):
        super(AnalogSGD_approx, self).__setstate__(state)

    # Performs a single optimization step for parameter updates
    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # param_groups-->a dict containing all parameter groups
        for group in self.param_groups:
            # Retrieving from defaults dictionary
            learn_rate = group['lr']
        #    factor = group['c']
            active_radius = group['active_radius']
            noise_std = group['noise_std']

           # Update step for each parameter present in param_groups
            for param in group['params']:
                # Calculating gradient('∇f(x,ε)' in paper)
                if param.grad is None:
                    continue
                grad = param.grad.data
                grad.add_(noise_std * torch.randn_like(grad))

                # Updation of model parameter p                
                # p.data = p.data-learn_rate*dp-(learn_rate/active_radiusi)*torch.abs(dp)*(p-self.symmetric_point)
                analog_increment(param, -learn_rate*grad, active_radius)
        return loss
def get_loss(model, for_training=True):
    outputs = model(A)
    loss = criterion(outputs.view(-1), Y)
    if for_training:
        return loss
    else:
        return loss / INPUT_SIZE
    
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

def get_model(rpu_config=None):
    if rpu_config is None:
        model = torch.nn.Linear(INPUT_SIZE, 1)
    else:
        model = AnalogLinear(INPUT_SIZE, 1, rpu_config=rpu_config)
    return model
def get_batch_loss(model, beg, batch_size=BATCH_SIZE):
    outputs = model(X_duplicated[beg:beg+batch_size])
    loss = criterion(outputs.view(-1), Y_duplicated[beg:beg+batch_size])
    return loss
def get_closure_loss_with_batch_idx(model, beg, batch_size=BATCH_SIZE):
    if beg is None:
        return None
    else:
        return lambda : get_batch_loss(model, beg, batch_size)
def get_device(tau):
    device = SoftBoundsReferenceDevice(construction_seed=10)
    device.dw_min = 1e-4
    device.dw_min_dtod = 0
    device.dw_min_std = 0
    device.write_noise_std = 0
    device.corrupt_devices_prob = 0
    device.corrupt_devices_range = 0
    device.up_down = 0
    device.up_down_dtod = 0
    device.reference_mean = 0
    device.reference_std = 0
    device.w_max = tau
    device.w_min = -tau
    device.w_min_dtod = 0
    device.w_max_dtod = 0
    return device
def get_IO(noise_std=0):
    io_param = IOParameters(
        is_perfect = False,
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
  
def run_AGD_approx(tau=4, noise_std=0.1, lr=0.1):
    # <<<<<<<<<<<<<<<<<< SHD <<<<<<<<<<<<<<<<<<<
    grad_per_iter = 1
    torch.manual_seed(618)
    # <<<<<<<< Define model <<<<<<<<
    model = get_model()
    init_model(model)
    # <<<<<<<< Define optimizer <<<<<<<<
    optimizer = AnalogSGD_approx(model.parameters(), 
                                lr=lr, noise_std=noise_std,
                                active_radius=tau)

    loss_list = []
    weight_list = []
    num_iteration = gradient_budget // grad_per_iter
    for iter_idx in range(num_iteration+1):
        # evaluate
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)
        weight = get_weights(model)
        weight_list.append(weight)
        model.train()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

    return loss_list, weight_list
def run_AGD(tau=4, noise_std=0.1, lr=0.1):
    rpu_config = SingleRPUConfig(
        device=get_device(tau=tau)
    )
    
    rpu_config.forward = get_IO(noise_std)
    rpu_config.backward = get_IO(noise_std)
    rpu_config.update.desired_bl = 800
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
        # evaluate
        # l = loss.detach().numpy()
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)
        weight = get_weights(model)
        weight_list.append(weight)
        # loss_list[ii].append(l-opt)
        model.train()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

    return loss_list, weight_list
def run_SGD(noise_std=0.1, lr=0.1):
    model = get_model(None)
    init_model(model)
    # print([p for p in model.parameters()])

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


import matplotlib.pyplot as plt

def get_asymptotic_err(loss_data):
    with torch.no_grad():
        last_20_losses = loss_data[-20:]
        avg_loss = sum(last_20_losses) / len(last_20_losses)
    return avg_loss
    
def plot_loss_curves(loss_collections, lr_values):
    SCALE = 1
    # SCALE = 0.7
    # plt.figure(figsize=(SCALE*3.7, SCALE*4))
    plt.figure(figsize=(SCALE*3.7, SCALE*2.8))

    show_opt_gap = args.opt_gap

    # >>>>>>>>>>>>>>>>> asymtotic learning error >>>>>>>>>>>>>>>>>
    for loss_collection in loss_collections:
        loss_data = loss_collection['data']
        asymptotic_errs = [
            get_asymptotic_err(loss_one_time)
            for loss_one_time in loss_data
        ]
        plt.plot(lr_values, asymptotic_errs, '-o', label=loss_collection['name'])

    plt.xlabel(r"Step Size $\alpha$", fontsize=12)
    plt.ylabel("Learning Error", fontsize=12)
    plt.legend(fontsize=10, loc="upper right")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    # plt.title("Average Loss vs Learning Rate", fontsize=14)

    # # 保存图像到文件
    pic_name = f'dynamic_loss_vs_lr'
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
    plt.clf()

    # >>>>>>>>>>>>>>>>> convergence >>>>>>>>>>>>>>>>>
    linestyles = ['-', '-']
    markers = ['*', 'o']
    for collection_idx, loss_collection in enumerate(loss_collections):
        loss_data = loss_collection['data']
        alg_name = loss_collection['name']
        linestyle = linestyles[collection_idx]
        marker = markers[collection_idx]
        for loss_idx, loss_values in enumerate(loss_data):
            label = fr'{alg_name} $\alpha$={lr_values[loss_idx]:.0e}'
            smooth_loss = []
            for loss in loss_values:
                if len(smooth_loss) == 0:
                    smooth_loss.append(loss.detach().item())
                else:
                    smooth_coef = 0.9
                    res = smooth_coef*smooth_loss[-1]+(1-smooth_coef)*loss
                    smooth_loss.append(res.detach().item())
            plt.plot(smooth_loss, 
                    #  label=label, 
                     markevery=300,
                     linestyle=linestyle, marker=marker, 
                     color=f'C{loss_idx}')

    # plt.ylim(top=2, bottom=0)
    # plt.title("Loss Curves")
    # plt.xlabel("Number of Gradient Computation")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(W_k)-f^*$", fontsize=10)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=f'C{lr_idx}', lw=4)
                    for lr_idx in range(len(lr_values))
        ] + [
            Line2D([0], [0], marker=markers[i], color=f'k', lw=0)
            for i in range(len(loss_collections))
        ]
    legend_labels = [
        fr'$\alpha$={lr_values[lr_idx]:.0e}' for lr_idx in range(len(lr_values))
        ] + [
            loss_collection['name'] for loss_collection in loss_collections
        ]
    plt.legend(custom_lines, legend_labels,
    #     # fontsize=18,
        bbox_to_anchor=(1,0), loc="lower left",
    )  
    plt.yscale('log')
    plt.grid(True)
    
    # pic_name = f'DEV={DEVICE_NAME}_{optimizer.__class__.__name__}_BS={BATCH_SIZE}'
    # pic_name = f'{optimizer.__class__.__name__}_BS={BATCH_SIZE}'
    pic_name = f'dynamic_loss_vs_lr_convergence'
    # file_dir = os.path.dirname(os.path.abspath(__file__))
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


tau = 3
noise_std = 1e-2 / math.sqrt(INPUT_SIZE)
# lr_values = [1, 0.5, 0.2, 0.1]

# actually_lrs = [0.2*lr for lr in lr_values]
actually_lrs = [2e-1, 1e-1, 5e-2, 3e-2]

collection_AGD_dynamic = {
    'name': 'Analog SGD',
    'data': [
        # run_AGD_approx(tau, noise_std, lr)[0]
        run_AGD(tau, noise_std, lr)[0]
        for lr in actually_lrs
    ]
}

collection_SGD = {
    'name': 'Digital SGD',
    'data': [
        run_SGD(noise_std, lr)[0]
        for lr in actually_lrs
    ]
}

# collection_AGD = {
#     'name': 'Analog GD',
#     'data': [
#         run_AGD(tau, noise_std, lr)[0]
#         for lr in actually_lrs
#     ]
# }

# collection_AGD = {
#     'name': 'Analog GD',
#     'data': [
#         run_AGD(tau, noise_std, lr)[0]
#         for lr in actually_lrs
#     ]
# }

loss_collections = [
    # collection_AGD, 
    collection_SGD,
    collection_AGD_dynamic,
]
plot_loss_curves(loss_collections, actually_lrs)
