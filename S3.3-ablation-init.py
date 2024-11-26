# %%
import math

import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.optim import Optimizer

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext

from aihwkit.optim import AnalogSGD

from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    SingleRPUConfig,
    WeightNoiseType,
)
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    LinearStepDevice,
    SoftBoundsReferenceDevice,
    TransferCompound,
)
from aihwkit.simulator.parameters import (
    IOParameters,
)
import argparse

parser = argparse.ArgumentParser(description="A simple command-line argument example")

# Add command line arguments
parser.add_argument('-BS', '--batch-size', type=int, help="", default=100)
parser.add_argument('-ALG', '--algorithm', type=str, help="", default='AnalogSGD')

# Parse the command line arguments
args = parser.parse_args()

KEY_INNER_ITERATION = 'inner iteration'
def analog_increment(tensor, increment, tau, weight_decay=1):
    with torch.no_grad():
        tensor.data.mul_(weight_decay).add_(increment - 1/tau*torch.abs(increment)*tensor)
    return tensor
class HamiltonianDescent(Optimizer):
    def __init__(self, params, alpha, beta, tau, update_frequency=1, noise_std=0.1):
        defaults = dict(alpha=alpha, beta=beta, tau=tau, 
                        update_frequency=update_frequency,
                        noise_std=noise_std,
                        momentum={})
        super(HamiltonianDescent, self).__init__(params, defaults)
        
        for group in self.param_groups:
           momentum = group['momentum']
           for p in group['params']:
                momentum.update({
                    p: torch.zeros_like(p) 
                })
        self.state[KEY_INNER_ITERATION] = 0

    def step(self, closure=None):
        state = self.state
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            learn_rate = group['alpha']
            beta = group['beta']
            tau = group['tau']
            update_frequency = group['update_frequency']
            momentum = group['momentum']
            noise_std = group['noise_std']

            for param in group['params']:
                param.data.add_(momentum[param], alpha=gamma)
                
                if param.grad is None:
                    continue
                grad = param.grad.data 
                grad.add_(noise_std * torch.randn_like(grad))
                # Update P
                analog_increment(momentum[param], beta*grad, tau)

                # Update the weights
                if state[KEY_INNER_ITERATION] % update_frequency == 0:
                    analog_increment(param, -learn_rate * momentum[param], tau)
                    
                param.data.add_(momentum[param], alpha=-gamma)
                    
        state[KEY_INNER_ITERATION] += 1
        if state[KEY_INNER_ITERATION] % update_frequency == 0:
            state[KEY_INNER_ITERATION] = 0

        return loss

class AnalogSGD_approx(Optimizer):
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
                analog_increment(param, -learn_rate*grad, active_radius)
        return loss


# lr = 0.1 is a baseline lr
lr = 0.1
fast_lr = 1e-1 * (lr / 0.1)
gamma = 0.

gradient_budget = 50 * int(0.1 / lr)
# gradient_budget = 100 * int(0.1 / lr)

INPUT_SIZE = 50
DATASET_SIZE = 100
torch.manual_seed(42)  # For reproducibility
element_scale = 0.5
x_true = element_scale * torch.randn(INPUT_SIZE, requires_grad=False)
# bias_true = element_scale * torch.randn(1, requires_grad=False)  # Noise term
bias_true = 0.1
print(x_true.numel())
print(x_true.norm(p=1) / x_true.numel())
A = torch.randn((DATASET_SIZE, INPUT_SIZE), requires_grad=False)
Y = torch.matmul(A, x_true) + bias_true

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
def init_model(model, init_value=0):
    for param in model.parameters():
        if isinstance(param, AnalogContext):
            tile = param.analog_tile
            w_z = init_value * torch.ones(tile.in_size, tile.out_size)
            b_z = init_value * torch.ones(tile.out_size)
            tile.set_weights(w_z, b_z)
        else:
            torch.nn.init.constant_(param, init_value)
def get_device_comprehensive(device_name, tau = 4):
    if device_name == 'CS':
        return ConstantStepDevice()
    elif device_name == 'SB':
        
        # return SoftBoundsDevice(w_max=tau, 
        #                         w_min=0, 
        #                         # w_min=-tau, 
        #                         construction_seed=10)
        return SoftBoundsReferenceDevice(
            w_max=tau, w_min=-tau, 
            w_max_dtod=0, w_min_dtod=0,
            # reference_mean=tau, reference_std=0,
            dw_min=1e-10,
            dw_min_dtod=0,
            dw_min_std=0,
            write_noise_std=0,
            subtract_symmetry_point=True,
            perfect_bias=True,
            enforce_consistency=True,
            construction_seed=10)
    elif device_name == 'LS':
        LinearStepDevice(w_max_dtod=0.4)
    else:
        raise NotImplemented
def get_device(device_name, tau):
    device = SoftBoundsReferenceDevice(construction_seed=10)
    device.dw_min = 1e-4
    # device.dw_min = 7e-8
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
def get_model(rpu_config=None):
    if rpu_config is None:
        model = torch.nn.Linear(INPUT_SIZE, 1, True)
    else:
        model = AnalogLinear(INPUT_SIZE, 1, True, rpu_config=rpu_config)
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
    loss = criterion(outputs.view(-1), Y)
    if for_training:
        return loss
    else:
        return loss / INPUT_SIZE
# def get_loss(model, for_training=True):
#     outputs = model(A)
#     loss = criterion(outputs.view(-1), Y)
#     if for_training:
#         return loss
#     else:
#         return loss / INPUT_SIZE

def get_IO(noise_std=0):
    io_param = IOParameters(
        is_perfect  = False,
        # is_perfect  = True,
        inp_bound  = 1,
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
    )
    return io_param
    
def run_SGD(noise_std=0.1):
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
    loss_recorded = get_loss(model, for_training=False)
    loss_list.append(loss_recorded)
    return loss_list, weight_list, model

def run_AGD_approx(tau=4, noise_std=0.1):
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
        # l = loss.detach().numpy()
        model.eval()
        loss = get_loss(model)
        loss_recorded = get_loss(model, for_training=False)
        loss_list.append(loss_recorded)
        # loss_list[ii].append(l-opt)
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

def run_SHD(tau=4, noise_std=0.1):
    # <<<<<<<<<<<<<<<<<< SHD <<<<<<<<<<<<<<<<<<<
    grad_per_iter = 1
    torch.manual_seed(618)
    # <<<<<<<< Define model <<<<<<<<
    model = get_model()
    init_model(model)
    # <<<<<<<< Define optimizer <<<<<<<<
    optimizer = HamiltonianDescent(model.parameters(), 
                                   alpha=lr, beta=fast_lr, 
                                #    alpha=lr, beta=5e-3, 
                                    noise_std=noise_std,
                                   tau=tau, update_frequency=1)

    loss_list = []
    weight_list = []
    num_iteration = gradient_budget // grad_per_iter
    for iter_idx in range(num_iteration+1):
        # evaluate
        # l = loss.detach().numpy()
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

            # print("Learned weights:", learned_weights)
            # print("Learned bias:", learned_bias)

        # print([param for param in model.parameters()])
        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # plt.plot(loss_list, label='SHD')
    return loss_list, weight_list

def run_TT(tau=4, noise_std=0.1):
    units_in_mbatch = True
    construction_seed = 23
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            # devices that compose the Tiki-taka compound
            unit_cell_devices=[
                get_device('SB', tau=tau),
                get_device('SB', tau=tau),
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
            # transfer_forward=get_IO(),
        )
    )
    
    rpu_config.forward = get_IO(noise_std=noise_std)
    rpu_config.backward = get_IO(noise_std=noise_std)
    # rpu_config.backward.out_noise = 1e-3
    # rpu_config.backward.inp_res = 1e-19
    # rpu_config.backward.out_res = 1e-19
    rpu_config.update.desired_bl = 300
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

        optimizer.step()
        optimizer.zero_grad()

        # print([param for param in model.parameters()])
        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # plt.plot(loss_list, label='SHD')
    return loss_list, weight_list

def run_AGD(tau=4, noise_std=0.1, initial_model=None):
    rpu_config = SingleRPUConfig(
        device=get_device('SB', tau=tau)
    )
    
    # rpu_config.forward = get_IO(noise_std)
    rpu_config.forward = get_IO(noise_std)
    rpu_config.backward = get_IO(noise_std)
    rpu_config.update.desired_bl = 400
    rpu_config.update.sto_round = True
    
    torch.manual_seed(618)
    # <<<<<<<< Define model <<<<<<<<
    model = get_model(rpu_config)
    if initial_model is None:
        init_model(model)
    else:
        for param, param_init in zip(model.parameters(), initial_model.parameters()):
            if isinstance(param, AnalogContext):
                tile = param.analog_tile
                tile.set_weights(param_init.data)
            else:
                param.data.copy_(param_init)
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
    # print(loss_list[0])

    return loss_list, weight_list



import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(loss_data, weight_data, begin_froms, labels, linestyles):
    """
    Plot multiple loss curves on the same graph.

    Parameters:
    - loss_data: A list of lists/tuples, where each inner list/tuple contains loss values.
    - labels: A list of labels corresponding to each loss curve.
    """
    SCALE = 0.7
    plt.figure(figsize=(SCALE*5, SCALE*4))  # Adjust the figure size as needed

    for idx, (loss_values, weight_values, label, linestyle) in enumerate(zip(loss_data, weight_data, labels, linestyles)):
        smooth_loss = []
        for loss in loss_values:
            if len(smooth_loss) == 0:
                smooth_loss.append(loss.detach().item())
            else:
                smooth_coef = 0.
                res = smooth_coef*smooth_loss[-1]+(1-smooth_coef)*loss
                smooth_loss.append(res.detach().item())

        # idx_list = [i + idx * gradient_budget for i in range(len(smooth_loss))]
        idx_list = [i + begin_froms[idx] for i in range(len(smooth_loss))]
        plt.plot(idx_list, smooth_loss, label=label, linestyle=linestyle)

    # plt.ylim(top=2, bottom=0)
    # plt.title("Loss Curves")
    plt.xlabel("Number of Gradient Computation (k)")
    plt.ylabel("$f(W_k)-f^*$")
    # plt.legend(
    #     # fontsize=18,
    #     # bbox_to_anchor=(1,0), loc="lower left",
    # )
    plt.yscale('log')
    plt.grid(True)
    
    # # plt.subplot(2, 1, 2)
    # for loss_values, weight_values, label, linestyle in zip(loss_data, weight_data, labels, linestyles):
    #     # weight_values = [w-2 for w in weight_values]
    #     plt.plot(weight_values, label=label, linestyle=linestyle)
    # plt.title("Weight Curves")
    # # plt.ylabel(r'$\nabla f(W_k)$')
    # plt.ylabel('$W_k$')
    # plt.xlabel("Number of Gradient Computation (k)")
    # plt.yscale('log')
    
    plt.subplots_adjust(hspace=0.6)
    # pic_name = f'DEV={DEVICE_NAME}_{optimizer.__class__.__name__}_BS={BATCH_SIZE}'
    # pic_name = f'{optimizer.__class__.__name__}_BS={BATCH_SIZE}'
    # pic_name = f'SHD-vs-TT' + f'-flr={fast_lr}-g={gamma}'
    # pic_name = f'alg-comparison-lr={lr}-noise-{noise_std:.2e}'
    pic_name = f'AGD-from-diff'
    # pic_name = 'A05-TT-exploration'
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
    # plt.show()

tau = 1
# noise_std = 0
noise_std = 1e-2 / math.sqrt(INPUT_SIZE)
# noise_std = 1e-2 / math.sqrt(INPUT_SIZE)

def get_better_model(iter_time):
    model = get_model(None)
    init_model(model)
    # print([p for p in model.parameters()])

    optimizer = optim.SGD(model.parameters(), lr=lr)  # Learning rate

    for epoch in range(iter_time):
        loss = get_loss(model)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model
def get_init_model(initial_points, iter_times):
    model = get_model(None)
    for initial_point in initial_points:
        init_model(model, init_value=initial_point)
        yield model
    for iter_time in iter_times:
        model = get_better_model(iter_time)
        yield model
    
    
initial_points = [1e3, 1e2, 1e1, 1e-0, 1e-2]
iter_times = [10, 20, 30]

num_curve = len(initial_points) + len(iter_times)
loss_lists = []
weight_lists = []
begin_froms = [0] * num_curve
labels = initial_points + iter_times
linestyles = ['-'] * num_curve
for initial_model in get_init_model(initial_points, iter_times):
    # init_model(digital_temp_model, init_value=initial_point)
    print(get_loss(initial_model, for_training=False))

    loss_list, weight_list = run_AGD(tau, noise_std, initial_model=initial_model)
    loss_lists.append(loss_list)
    weight_lists.append(weight_list)

# for alg, loss_values in zip(labels, loss_lists):
#     print(alg, loss_values[-1].item())
plot_loss_curves(loss_lists, weight_list, begin_froms, labels, linestyles)
