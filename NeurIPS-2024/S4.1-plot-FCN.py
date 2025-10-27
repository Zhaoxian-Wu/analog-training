import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.lines import Line2D

MAX_EPOCH = 80

def read_tensorboard_data(logdir, tag, max_step=-1):
    # Create EventAccumulator object and reload event files
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Initialize data dictionary
    dict_step_value = {}
    
    # Traverse all scalar events
    for scalar_event in event_acc.Scalars(tag):
        step = scalar_event.step
        value = scalar_event.value
        
        if max_step > 0 and step > max_step:
            continue
        
        # Categorize data into different runs based on steps
        if step not in dict_step_value:
            dict_step_value[step] = []
        dict_step_value[step].append(value)
        
    data = {'steps': [], 'values_avg': [], 'values_var': [], 'values_std': [], 'repeat_time': None}
    # Check if the number of data points in each run is consistent
    unique_counts = set([len(values) for values in dict_step_value.values()])
    if len(unique_counts) == 1:
        data['repeat_time'] = list(unique_counts)[0]
    else:
        inconsistent_steps = [step for step, values in dict_step_value.items() if len(values) != max(unique_counts)]
        print(f"[Warning] {logdir}: Missing some data (Steps: {min(inconsistent_steps)}--{max(inconsistent_steps)})")
        data['repeat_time'] = max(unique_counts)
    
    # Calculate mean, variance, and standard deviation for each step
    for step, values in dict_step_value.items():
        data['steps'].append(step)
        data['values_avg'].append(np.mean(values))
        data['values_var'].append(np.var(values))
        data['values_std'].append(np.std(values))
    
    return data

def plot_tensorboard_data(data_list, tag, save_path_png, save_path_pdf, 
                          legend_lines, legend_text):
    SCALE = 0.6
    fig = plt.figure(figsize=(SCALE*6.4, SCALE*4.8))
    
    ax = plt.gca()

    axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                    bbox_to_anchor=(0.54, 0.20, 1, 1),
                    bbox_transform=ax.transAxes)

    for i, data in enumerate(data_list):
        linestyle = data['linestyle']
        marker = data['marker']
        color = data['color']
        running_data = data['data']
        
        steps = running_data['steps']
        
        smooth_loss = []
        for loss in running_data['values_avg']:
            if len(smooth_loss) == 0:
                smooth_loss.append(loss)
            else:
                gamma = 0.8
                smooth_loss.append(gamma*smooth_loss[-1]+(1-gamma)*loss)
                
        smooth_loss = [100*v for v in smooth_loss]
        ax.plot(steps, smooth_loss, 
                 linestyle=linestyle, marker=marker, color=color,
                 markevery=5)
        axins.plot(steps, smooth_loss, 
                 linestyle=linestyle, marker=marker, color=color,
                #  markersize=7,
                 markevery=1)
    
    # Adjust display range of the sub-graph 
    axins.set_xlim(77, 80)
    axins.set_ylim(97, 97.8)
    axins.grid(True)
    # axins.set_xticks([])
    # axins.set_yticks([])

    # Establish connection lines between the main and inset axes
    # loc1 loc2: four corners of the axes
    # 1 (upper right) 2 (upper left) 3 (lower left) 4 (lower right)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)

    ax.set_ylim(bottom=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    # ax.grid(True)
    ax.legend(legend_lines, legend_text, 
        bbox_to_anchor=(1,0), loc="lower left",
    )

    plt.savefig(save_path_png, format='png', bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    
    print(f'finish: {save_path_pdf}')

    plt.clf()

if __name__ == "__main__":
    log_directory = 'runs/MNIST-FCN'
    tensorboard_tag = 'Accuracy/test'
    
    subdirectories = []
    
    data_list = []
    
    taus = [0.7, 0.78, 0.8]
    legend_lines = [
        Line2D([0], [0], color=f'C{tau_idx}', lw=4)
        for tau_idx in range(len(taus))
    ]
    legend_text = [
        fr'$\tau={tau}$' for tau in taus
    ]
    
    for tau_idx, tau in enumerate(taus):
        subdir_path = os.path.join(log_directory, f'Analog SGD-tau={tau}')
        tensorboard_data = read_tensorboard_data(subdir_path, tensorboard_tag, max_step=MAX_EPOCH)
        data_list.append({
            'data': tensorboard_data,
            'linestyle': '--',
            'marker': 'o',
            'color': f'C{tau_idx}'
        })
        mean_final = sum(tensorboard_data['values_avg'][-5:])*100/5
        stdv_final = sum(tensorboard_data['values_std'][-5:])*100/5
        print(fr'Analog SGD-tau={tau}: {mean_final:.2f} \stdv{{$\pm$ {stdv_final:.2f}}} steps: {len(tensorboard_data["steps"])} repeat: {tensorboard_data["repeat_time"]}')
    legend_lines.append(
        Line2D([0], [0], linestyle=data_list[-1]['linestyle'],
               marker=data_list[-1]['marker'],
               markersize=4,
               color=f'k')
    )
    legend_text.append('Analog SGD')
    
    for tau_idx, tau in enumerate(taus):
        subdir_path = os.path.join(log_directory, f'TT-v1-tau={tau}')
        tensorboard_data = read_tensorboard_data(subdir_path, tensorboard_tag, max_step=MAX_EPOCH)
        data_list.append({
            'data': tensorboard_data,
            'linestyle': '-',
            'marker': '^',
            'color': f'C{tau_idx}'
        })
        mean_final = sum(tensorboard_data['values_avg'][-5:])*100/5
        stdv_final = sum(tensorboard_data['values_std'][-5:])*100/5
        print(fr'TT-tau={tau}: {mean_final:.2f} \stdv{{$\pm$ {stdv_final:.2f}}} steps: {len(tensorboard_data["steps"])} repeat: {tensorboard_data["repeat_time"]}')
    legend_lines.append(
        Line2D([0], [0], linestyle=data_list[-1]['linestyle'],
               marker=data_list[-1]['marker'],
               markersize=4,
               color=f'k')
    )
    legend_text.append('Tiki-Taka')
    
    # subdir_path = os.path.join(log_directory, f'mp')
    # tensorboard_data = read_tensorboard_data(subdir_path, tensorboard_tag, max_step=MAX_EPOCH)
    # data_list.append({
    #     'data': tensorboard_data,
    #     'linestyle': '-',
    #     'marker': 'D',
    #     'color': f'C{len(taus)+1}'
    # })
    # print(f'MP: {sum(tensorboard_data["values"][-5:])*100/5:.2f} steps: {len(tensorboard_data["steps"])}')
    # legend_lines.append(
    #     Line2D([0], [0], linestyle=data_list[-1]['linestyle'],
    #            marker=data_list[-1]['marker'], 
    #            color=data_list[-1]['color'], 
    #         )
    # )
    # legend_text.append(r'MP ($\tau=0.7$)')
    
    subdir_path = os.path.join(log_directory, f'FP SGD')
    tensorboard_data = read_tensorboard_data(subdir_path, tensorboard_tag, max_step=MAX_EPOCH)
    data_list.append({
        'data': tensorboard_data,
        'linestyle': '-.',
        'marker': 's',
        'color': f'C{len(taus)}'
    })
    legend_lines.append(
        Line2D([0], [0], linestyle=data_list[-1]['linestyle'],
               marker=data_list[-1]['marker'], 
               color=data_list[-1]['color'], 
            )
    )
    mean_final = sum(tensorboard_data['values_avg'][-5:])*100/5
    stdv_final = sum(tensorboard_data['values_std'][-5:])*100/5
    print(fr'FP: {mean_final:.2f} \stdv{{$\pm$ {stdv_final:.2f}}} steps: {len(tensorboard_data["steps"])} repeat: {tensorboard_data["repeat_time"]}')
    legend_text.append('Digital SGD')
    
    # linestyle_markers = [
    #     ('-', 'o', 'blue'),
    #     ('--', '^', 'orange'),
    #     ('-.', 's', 'green'),
    #     (':', 'D', 'red')
    # ]
    
    # Generate a separate plot file for each folder
    file_dir = ''
    dir_png_path = os.path.join(file_dir, 'fig', 'png')
    dir_pdf_path = os.path.join(file_dir, 'fig', 'pdf')
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)
    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    pic_png_path = os.path.join(dir_png_path, 'A03-MNIST-FCN.png')
    pic_pdf_path = os.path.join(dir_pdf_path, 'A03-MNIST-FCN.pdf')

    # Plot and save the curve
    plot_tensorboard_data(data_list, tensorboard_tag, 
                          pic_png_path, pic_pdf_path, 
                          legend_lines, legend_text)
