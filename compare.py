import matplotlib.pyplot as plt
import numpy as np

def calculate_smooth_and_variance(data, window_size):
    """
    Calculates the smoothed data and variance for the given data using a moving window.
    """
    new_data = []
    variances = []
    for i in range(len(data)):
        if i < window_size:
            variances.append(np.sqrt(np.var(data[:i+1])))
            new_data.append(sum(data[:i+1]) / (i+1))
        else:
            variances.append(np.sqrt(np.var(data[i-window_size:i])))
            new_data.append(sum(data[i-window_size:i]) / window_size)
    return new_data, variances

def plot_comparison(all_data, colors, filename, window_size=150):
    max_epoch = 0
    for name in all_data.keys():
        data = all_data[name]
        if len(data) > max_epoch:
            max_epoch = len(data)
    
    i = 0
    for name in all_data.keys():
        data = all_data[name]
        data = calculate_smooth_and_variance(data, window_size=window_size)[0]
        x_ticks = np.arange(len(data)) * max_epoch / len(data)
        plt.plot(x_ticks, data, label=name, color=colors[i])
        i += 1

    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Comparison of Scores')
    plt.legend()
    plt.grid()
    plt.savefig(filename, dpi=600)
    plt.close('all')

def plot_variance(all_data, colors, filename, window_size=150):
    max_epoch = 0
    max_score = -10000
    min_score = 10000
    for name in all_data.keys():
        data = all_data[name]
        if len(data) > max_epoch:
            max_epoch = len(data)
        if max(data) > max_score:
            max_score = max(data)
        if min(data) < min_score:
            min_score = min(data)
    
    plt.subplots_adjust(hspace=0.5)
    i = 0
    for name in all_data.keys():
        plt.subplot(len(all_data), 1, i+1)
        data = all_data[name]
        data, variances = calculate_smooth_and_variance(data, window_size=window_size)
        x_ticks = np.arange(len(data)) * max_epoch / len(data)
        plt.plot(x_ticks, data, label=name, color=colors[i])
        plt.fill_between(x_ticks, np.array(data) - np.array(variances), np.array(data) + np.array(variances), color=colors[i], alpha=0.2)
        i += 1
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(name, fontsize=40)
        plt.grid()
        plt.ylim(min_score - 50, max_score + 50)
    plt.savefig(filename, dpi=600)
    plt.close('all')

if __name__ == "__main__":
    # plt.rcParams['font.size'] = 18
    # plt.rcParams['figure.figsize'] = (10, 6)
    map_name = 'map02'
    colors = [[0, 0, 1], [0.5, 0, 0], [0, 1, 0], [0.6, 0.6, 0], [0, 0.6, 0.6], [0.6, 0, 0.6]]
    for prefix in ['', 'dueling', 'double_dueling', 'double']: # , 'dueling', 'double_dueling', 'double'
        for postfix in ['']:
            all_data = {}
            if prefix == '':
                for name in ['basic', 'double', 'dueling', 'double_dueling']:
                    name = prefix + name
                    data_path = f'./results/Deep_Q-learning_{name}_{map_name}{postfix}/all_results.npy'
                    all_data[name] = np.load(data_path)
                file_name = f'comparison_plot_{map_name}{postfix}.png'
            elif map_name == 'map02':
                for name in ['', '_resolution', '_sgd', '_transformer', '_reward', '_sgd_transformer', '_resolution_transformer']:
                    name = prefix + name
                    data_path = f'./results/Deep_Q-learning_{name}_{map_name}{postfix}/all_results.npy'
                    try:
                        all_data[name] = np.load(data_path)
                    except:
                        pass
                file_name = f'comparison_plot_{prefix}_{map_name}{postfix}.png'
            if len(all_data) > 0:
                plt.rcParams['font.size'] = 18
                plt.rcParams['figure.figsize'] = (10, 6)
                plot_comparison(all_data, colors, file_name, 300)
                plt.rcParams['font.size'] = 24
                plt.rcParams['figure.figsize'] = (12, int(len(all_data) * 6))
                plot_variance(all_data, colors, file_name.replace('comparison','variance'), 20)
    
    all_data = {}
    for name in ['basic', 'double_transformer', 'dueling_resolution', 'double_dueling_sgd_transformer']:
        data_path = f'./results/Deep_Q-learning_{name}_{map_name}/all_results.npy'
        all_data[name] = np.load(data_path)
    file_name = f'comparison_plot_improved_{map_name}.png'
    plt.rcParams['font.size'] = 18
    plt.rcParams['figure.figsize'] = (10, 6)
    plot_comparison(all_data, colors, file_name, 300)
    plt.rcParams['font.size'] = 24
    plt.rcParams['figure.figsize'] = (12, int(len(all_data) * 6))
    plot_variance(all_data, colors, file_name.replace('comparison','variance'), 20)