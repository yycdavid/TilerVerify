import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import h5py


def get_error_matrix(errors, points_per_grid, offset_grid_num, angle_grid_num):
    # Reshape
    max_errors = np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)
    return np.reshape(max_errors, (offset_grid_num, angle_grid_num))

def plot_heat_map(error_matrix, target_name, type, offset_range, angle_range, result_dir, capped=False):
    if target_name == 'angle':
        measurement_range = 120
    else:
        measurement_range = 80

    color_max = measurement_range * 0.03

    plt.figure()
    #sns.set()
    ytick_space = offset_range*2//10
    xtick_space = angle_range*2//10
    offset_range = [-offset_range, offset_range]
    angle_range = [-angle_range, angle_range]
    yticks = np.arange(offset_range[0], offset_range[1]+1, ytick_space)
    xticks = np.arange(angle_range[0], angle_range[1]+1, xtick_space)

    plt.rcParams["axes.grid"] = False
    plt.imshow(error_matrix, cmap='rainbow')
    cb = plt.colorbar(extend='max')
    plt.clim(0, color_max);
    ax = plt.gca()
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)

    if target_name == 'angle':
        #ax = sns.heatmap(error_matrix, center=0, cmap=sns.diverging_palette(240, 0, s=99, l=50, as_cmap=True), square=True, xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': 'Degree'})
        #ax = sns.heatmap(error_matrix, center=0, cmap=cm.get_cmap(name='rainbow'), square=True, xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': 'Degree'})
        cb.set_label('Degree')
    else:
        #ax = sns.heatmap(error_matrix, center=0, cmap=sns.diverging_palette(240, 0, s=99, l=50, as_cmap=True), square=True, xticklabels=xticks, yticklabels=yticks, cbar_kws={'label': 'Length unit'})
        cb.set_label('Length unit')
    x_tick_position = (xticks - angle_range[0])/(angle_range[1] - angle_range[0])*ax.get_xlim()[1]
    y_tick_position = (yticks - offset_range[0])/(offset_range[1] - offset_range[0])*ax.get_ylim()[0]
    ax.set_xticks(x_tick_position)
    ax.set_yticks(y_tick_position)
    ax.set(xlabel='Angle', ylabel='Offset')
    ax.set(title='Error ' + type + ' map for '+target_name)
    if not capped:
        plot_file_path = os.path.join(result_dir, target_name+"_"+ type + ".png")
    else:
        plot_file_path = os.path.join(result_dir, target_name+"_"+ type + "_capped" + ".png")
    plt.savefig(plot_file_path)

def cap_matrix(bound_matrix, est_matrix, max_value):
    flatten_bound = np.reshape(bound_matrix, -1)
    flatten_est = np.reshape(est_matrix, -1)
    mask = flatten_bound > max_value
    flatten_bound[mask] = -5
    flatten_est[mask] = -5
    return np.reshape(flatten_bound, bound_matrix.shape), np.reshape(flatten_est, est_matrix .shape)

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
    offset_error_est_matrix = get_error_matrix(error_est_result['offset_errors'], points_per_grid, offset_grid_num, angle_grid_num)
    angle_error_est_matrix = get_error_matrix(error_est_result['angle_errors'], points_per_grid, offset_grid_num, angle_grid_num)

    plots_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(plots_dir):
        print("Creating {}".format(plots_dir))
        os.makedirs(plots_dir)
    # Plot heatmap
    plot_heat_map(offset_error_est_matrix, 'offset', 'estimate', args.offset_range, args.angle_range, plots_dir)
    plot_heat_map(angle_error_est_matrix, 'angle', 'estimate', args.offset_range, args.angle_range, plots_dir)

    # Plot error bound maps
    error_bound_file_path = os.path.join(result_dir, 'error_bound_result.mat')
    with h5py.File(error_bound_file_path, 'r') as f:
        error_bound_result = {}
        for k, v in f.items():
            error_bound_result[k] = np.array(v)
    offset_grid_num = int(error_bound_result['offset_grid_num'])
    angle_grid_num = int(error_bound_result['angle_grid_num'])
    offset_error_bound_matrix = np.reshape(error_bound_result['offset_errors'], (offset_grid_num, angle_grid_num))
    angle_error_bound_matrix = np.reshape(error_bound_result['angle_errors'], (offset_grid_num, angle_grid_num))
    plot_heat_map(offset_error_bound_matrix, 'offset', 'bound', args.offset_range, args.angle_range, plots_dir)
    plot_heat_map(angle_error_bound_matrix, 'angle', 'bound', args.offset_range, args.angle_range, plots_dir)

    # Plot error gap maps
    plot_heat_map(offset_error_bound_matrix - offset_error_est_matrix, 'offset', 'gap', args.offset_range, args.angle_range, plots_dir)
    plot_heat_map(angle_error_bound_matrix - angle_error_est_matrix, 'angle', 'gap', args.offset_range, args.angle_range, plots_dir)

    ## Obtain capped error matrices
    #max_error = 5
    #capped_offset_bound, capped_offset_estimate = cap_matrix(offset_error_bound_matrix, offset_error_est_matrix, max_error)
    #plot_heat_map(capped_offset_bound, 'offset', 'bound', args.offset_range, args.angle_range, plots_dir, capped=True)
    #plot_heat_map(capped_offset_estimate, 'offset', 'estimate', args.offset_range, args.angle_range, plots_dir, capped=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
