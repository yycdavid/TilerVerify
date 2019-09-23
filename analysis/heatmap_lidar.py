import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import h5py
import matplotlib.colors as colors

def get_error_matrix(errors, points_per_grid, distance_grid_num, angle_grid_num):
    # Reshape
    max_errors = np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)
    return np.reshape(max_errors, (distance_grid_num, angle_grid_num))

def plot_heat_map(verify_matrix, label, distance_min, distance_max, angle_range, result_dir):
    # verify_matrix: (angle_grid_num, distance_grid_num)
    angle_grid_num, distance_grid_num = verify_matrix.shape
    distance_range = [distance_min, distance_max]
    angle_range = [-angle_range, angle_range]

    # Compute distance bounds
    d_inv_start = 1.0 / distance_range[1]
    d_inv_end = 1.0 / distance_range[0]
    d_inv_step = (d_inv_end - d_inv_start) / distance_grid_num
    d_inv_bounds = np.arange(d_inv_start, d_inv_end + 1e-5, d_inv_step)
    d_bounds = np.flip(1.0/d_inv_bounds)
    assert len(d_bounds) == distance_grid_num + 1, "distance bounds count incorrect"

    # Angle bounds
    a_step = (angle_range[1] - angle_range[0]) / angle_grid_num
    a_bounds = np.arange(angle_range[0], angle_range[1]+1e-5, a_step)
    assert len(a_bounds) == angle_grid_num + 1, "angle bounds count incorrect"


    plt.figure()
    #sns.set()
    X,Y = np.meshgrid(a_bounds,d_bounds, indexing='ij')
    # Display matrix
    cmap = colors.ListedColormap(['green', 'red', 'orange'])
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.pcolormesh(X,Y,verify_matrix, cmap=cmap, norm=norm)

    ax = plt.gca()
    ax.set(xlabel='Angle', ylabel='Distance')

    plot_file_path = os.path.join(result_dir, "label_"+ str(label) + ".pdf")
    plt.savefig(plot_file_path, bbox_inches='tight')

    '''ytick_space = distance_range*2//5
    xtick_space = angle_range*2//5
    yticks = np.arange(distance_range[0], distance_range[1]+1, ytick_space)
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
    y_tick_position = (yticks - distance_range[0])/(distance_range[1] - distance_range[0])*ax.get_ylim()[0]
    ax.set_xticks(x_tick_position)
    ax.set_yticks(y_tick_position)
    ax.set(xlabel='Angle', ylabel='distance')
    ax.set(title=target_name + ' error ' + type)'''


def cap_matrix(bound_matrix, est_matrix, max_value):
    flatten_bound = np.reshape(bound_matrix, -1)
    flatten_est = np.reshape(est_matrix, -1)
    mask = flatten_bound > max_value
    flatten_bound[mask] = -5
    flatten_est[mask] = -5
    return np.reshape(flatten_bound, bound_matrix.shape), np.reshape(flatten_est, est_matrix .shape)

def main():
    parser = argparse.ArgumentParser(description='Get heatmap for verify results')
    parser.add_argument('--result_dir', help='path to result to get heatmap')
    parser.add_argument('--distance_min', type=int, help='range for distance')
    parser.add_argument('--distance_max', type=int, help='range for distance')
    parser.add_argument('--angle_range', type=int, help='range for angle')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

    # Plot error bound maps
    verify_file_path = os.path.join(result_dir, 'verify_result.mat')
    with h5py.File(verify_file_path, 'r') as f:
        verify_result = {}
        for k, v in f.items():
            verify_result[k] = np.array(v)

    distance_grid_num = int(verify_result['distance_grid_num'])
    angle_grid_num = int(verify_result['angle_grid_num'])

    plots_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(plots_dir):
        print("Creating {}".format(plots_dir))
        os.makedirs(plots_dir)

    for label in range(3):
        verify_matrix = np.reshape(verify_result['VerifyStatus_'+str(label)], (angle_grid_num, distance_grid_num))
        plot_heat_map(verify_matrix, label, args.distance_min, args.distance_max, args.angle_range, plots_dir)

    ## Obtain capped error matrices
    #max_error = 5
    #capped_distance_bound, capped_distance_estimate = cap_matrix(distance_verify_matrix, distance_error_est_matrix, max_error)
    #plot_heat_map(capped_distance_bound, 'distance', 'bound', args.distance_range, args.angle_range, plots_dir, capped=True)
    #plot_heat_map(capped_distance_estimate, 'distance', 'estimate', args.distance_range, args.angle_range, plots_dir, capped=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
