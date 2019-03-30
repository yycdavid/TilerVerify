import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


OFFSET_RANGE = [-10, 10]
ANGLE_RANGE = [-10, 10]

def get_error_matrix(errors, points_per_grid, offset_grid_num, angle_grid_num):
    # Reshape
    max_errors = np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)
    return np.reshape(max_errors, (offset_grid_num, angle_grid_num))

def plot_heat_map(error_matrix, target_name, type, offset_range, angle_range, result_dir):
    plt.figure()
    sns.set()
    ytick_space = offset_range*2//10
    xtick_space = angle_range*2//10
    offset_range = [-offset_range, offset_range]
    angle_range = [-angle_range, angle_range]
    yticks = np.arange(offset_range[0], offset_range[1]+1, ytick_space)
    xticks = np.arange(angle_range[0], angle_range[1]+1, xtick_space)

    ax = sns.heatmap(error_matrix, square=True, xticklabels=xticks, yticklabels=yticks)
    x_tick_position = (xticks - angle_range[0])/(angle_range[1] - angle_range[0])*ax.get_xlim()[1]
    y_tick_position = (yticks - offset_range[0])/(offset_range[1] - offset_range[0])*ax.get_ylim()[0]
    ax.set_xticks(x_tick_position)
    ax.set_yticks(y_tick_position)
    ax.set(xlabel='angle', ylabel='offset')
    ax.set(title='Error ' + type + ' map for '+target_name)
    plot_file_path = os.path.join(result_dir, target_name+"_"+ type + ".png")
    plt.savefig(plot_file_path)


def main():
    parser = argparse.ArgumentParser(description='Get heatmap for error')
    parser.add_argument('--result_dir', help='path to result to get heatmap')
    parser.add_argument('--offset_range', type=int, help='range for offset')
    parser.add_argument('--angle_range', type=int, help='range for angle')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

    # Plot error estimate maps
    error_est_file_path = os.path.join(result_dir, 'error_est_result.mat')
    error_est_result = sio.loadmat(error_est_file_path)

    # Compute max error for each grid
    points_per_grid = int(error_est_result['points_per_grid'])
    offset_grid_num = int(error_est_result['offset_grid_num'])
    angle_grid_num = int(error_est_result['angle_grid_num'])
    offset_error_matrix = get_error_matrix(error_est_result['offset_errors'], points_per_grid, offset_grid_num, angle_grid_num)
    angle_error_matrix = get_error_matrix(error_est_result['angle_errors'], points_per_grid, offset_grid_num, angle_grid_num)

    plots_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(plots_dir):
        print("Creating {}".format(plots_dir))
        os.makedirs(plots_dir)
    # Plot heatmap
    plot_heat_map(offset_error_matrix, 'offset', 'estimate', args.offset_range, args.angle_range, plots_dir)
    plot_heat_map(angle_error_matrix, 'angle', 'estimate', args.offset_range, args.angle_range, plots_dir)

    # Plot error bound maps
    error_bound_file_path = os.path.join(result_dir, 'error_bound_result.mat')
    with h5py.File(error_bound_file_path, 'r') as f:
        error_bound_result = {}
        for k, v in f.items():
            error_bound_result[k] = np.array(v)
    offset_grid_num = int(error_bound_result['offset_grid_num'])
    angle_grid_num = int(error_bound_result['angle_grid_num'])
    offset_error_matrix = np.reshape(error_bound_result['offset_errors'], (offset_grid_num, angle_grid_num))
    angle_error_matrix = np.reshape(error_bound_result['angle_errors'], (offset_grid_num, angle_grid_num))
    plot_heat_map(offset_error_matrix, 'offset', 'bound', args.offset_range, args.angle_range, plots_dir)
    plot_heat_map(angle_error_matrix, 'angle', 'bound', args.offset_range, args.angle_range, plots_dir)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
