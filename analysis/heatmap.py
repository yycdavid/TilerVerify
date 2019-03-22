import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

RESULTS_ROOT = 'trained_models'

OFFSET_RANGE = [-10, 10]
ANGLE_RANGE = [-10, 10]

def get_error_matrix(errors, points_per_grid, offset_grid_num, angle_grid_num):
    # Reshape
    max_errors = np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)
    return np.reshape(max_errors, (offset_grid_num, angle_grid_num))

def plot_heat_map(error_matrix, target_name):
    sns.set()
    offset_range = OFFSET_RANGE
    angle_range = ANGLE_RANGE
    yticks = np.arange(offset_range[0], offset_range[1]+2, 4)
    xticks = np.arange(angle_range[0], angle_range[1]+2, 2)

    ax = sns.heatmap(error_matrix, square=True, xticklabels=xticks, yticklabels=yticks)
    x_tick_position = (xticks - angle_range[0])/(angle_range[1] - angle_range[0])*ax.get_xlim()[1]
    y_tick_position = (yticks - offset_range[0])/(offset_range[1] - offset_range[0])*ax.get_ylim()[0]
    ax.set_xticks(x_tick_position)
    ax.set_yticks(y_tick_position)
    ax.set(xlabel='angle', ylabel='offset')
    ax.set(title='Error map for '+target_name)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Get heatmap for empirical error')
    parser.add_argument('--exp_name', help='name of experiment to get heatmap')
    parser.add_argument('--type', choices=['estimate', 'bound'])
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, args.exp_name)
    if args.type == 'estimate':
        error_file_path = os.path.join(exp_dir, 'error_est_result.mat')
        error_result = sio.loadmat(error_file_path)

        # Compute max error for each grid
        points_per_grid = int(error_result['points_per_grid'])
        offset_grid_num = int(error_result['offset_grid_num'])
        angle_grid_num = int(error_result['angle_grid_num'])
        offset_error_matrix = get_error_matrix(error_result['offset_errors'], points_per_grid, offset_grid_num, angle_grid_num)
        angle_error_matrix = get_error_matrix(error_result['angle_errors'], points_per_grid, offset_grid_num, angle_grid_num)
        # Plot heatmap
        plot_heat_map(offset_error_matrix, 'offset')
        #plot_heat_map(angle_error_matrix, 'angle')
    else:
        error_file_path = os.path.join(exp_dir, 'error_bound_result.mat')
        with h5py.File(error_file_path, 'r') as f:
            error_result = {}
            for k, v in f.items():
                error_result[k] = np.array(v)
        offset_grid_num = int(error_result['offset_grid_num'])
        angle_grid_num = int(error_result['angle_grid_num'])
        offset_error_matrix = np.reshape(error_result['offset_errors'], (offset_grid_num, angle_grid_num))
        angle_error_matrix = np.reshape(error_result['angle_errors'], (offset_grid_num, angle_grid_num))
        plot_heat_map(offset_error_matrix, 'offset')
        #plot_heat_map(angle_error_matrix, 'angle')


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
