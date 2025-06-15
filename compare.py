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

def plot_comparison(all_data, colors, filename):
    max_epoch = 0
    for name in all_data.keys():
        data = all_data[name]
        if len(data) > max_epoch:
            max_epoch = len(data)
    
    i = 0
    for name in all_data.keys():
        data = all_data[name]
        data = calculate_smooth_and_variance(data, window_size=150)[0]
        x_ticks = np.arange(len(data)) * max_epoch / len(data)
        plt.plot(x_ticks, data, label=name, color=colors[i])
        i += 1

    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Comparison of Scores')
    plt.legend()
    plt.grid()
    plt.savefig(filename, dpi=600)

if __name__ == "__main__":
    plt.rcParams['font.size'] = 14
    map_name = 'map02'
    all_data = {}
    colors = [[0, 0, 1], [0.5, 0, 0], [0, 1, 0], [0.6, 0.6, 0]]  # Blue, Red, Green, Orange
    for name in ['basic', 'double', 'dueling', 'double_dueling']:
        data_path = f'./results/Deep_Q-learning_{name}_{map_name}/all_results.npy'
        all_data[name] = np.load(data_path)
    file_name = f'comparison_plot_{map_name}.png'
    
    plot_comparison(all_data, colors, file_name)