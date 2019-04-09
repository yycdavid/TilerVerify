import os
import numpy as np
import argparse
import scipy.io as sio
import h5py
from heatmap import get_error_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


RESULTS_ROOT = 'trained_models'

def get_max_error_in_each_grid(errors, points_per_grid):
    return np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)

def get_bound_to_actual_ratio(bound, actual):
    return np.mean(np.divide(bound, actual))

def get_global_max(bound_matrix, est_matrix):
    flatten_bound = np.reshape(bound_matrix, -1)
    flatten_est = np.reshape(est_matrix, -1)
    return np.max(flatten_bound), np.max(flatten_est)

def get_stats_about_masked_region(bound_matrix, est_matrix, cap_value):
    # Compute 1) mask based on bound_matrix and cap_value
    flatten_bound = np.reshape(bound_matrix, -1)
    flatten_est = np.reshape(est_matrix, -1)
    mask_inside = flatten_bound > cap_value
    mask_outside = np.logical_not(mask_inside)
    percent_trusted = np.count_nonzero(mask_outside)/len(flatten_bound)
    # 2) Max error in estimate outside the mask, return
    max_est_outside = np.max(flatten_est[mask_outside])
    avg_est_outside = np.mean(flatten_est[mask_outside])
    max_bound_outside = np.max(flatten_bound[mask_outside])
    # 3) Average error in estimate inside the mask, return
    max_est_inside = np.max(flatten_est[mask_inside])
    avg_est_inside = np.mean(flatten_est[mask_inside])
    # 4) Percentage inside the mask that is bigger than max estimate error outside mask, return
    percent_bad_inside = np.count_nonzero(flatten_est[mask_inside] > max_est_outside)/len(flatten_est[mask_inside])
    percent_good_inside = np.count_nonzero(flatten_est[mask_inside] < avg_est_outside)/len(flatten_est[mask_inside])
    return percent_trusted, max_est_outside, avg_est_outside, max_est_inside, avg_est_inside, percent_bad_inside, percent_good_inside, max_bound_outside

def read_bound_est_result(result_dir):
    # Get estimate results
    error_est_file_path = os.path.join(result_dir, 'error_est_result.mat')
    error_est_result = sio.loadmat(error_est_file_path)
    points_per_grid = int(error_est_result['points_per_grid'])
    offset_grid_num = int(error_est_result['offset_grid_num'])
    angle_grid_num = int(error_est_result['angle_grid_num'])
    offset_error_est_matrix = get_error_matrix(error_est_result['offset_errors'], points_per_grid, offset_grid_num, angle_grid_num)
    angle_error_est_matrix = get_error_matrix(error_est_result['angle_errors'], points_per_grid, offset_grid_num, angle_grid_num)

    # Get bound results
    error_bound_file_path = os.path.join(result_dir, 'error_bound_result.mat')
    with h5py.File(error_bound_file_path, 'r') as f:
        error_bound_result = {}
        for k, v in f.items():
            error_bound_result[k] = np.array(v)
    offset_error_bound_matrix = np.reshape(error_bound_result['offset_errors'], (offset_grid_num, angle_grid_num))
    angle_error_bound_matrix = np.reshape(error_bound_result['angle_errors'], (offset_grid_num, angle_grid_num))

    return offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix

def compute_and_store_stats(result_dir, offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix):
    # Directory to store stats
    stats_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(stats_dir):
        print("Creating {}".format(stats_dir))
        os.makedirs(stats_dir)

    cap_value = 5
    percent_trusted, max_est_outside, avg_est_outside, max_est_inside, avg_est_inside, percent_bad_inside, percent_good_inside, max_bound_outside = get_stats_about_masked_region(offset_error_bound_matrix, offset_error_est_matrix, cap_value)

    angle_global_bound, angle_global_estimate = get_global_max(angle_error_bound_matrix, angle_error_est_matrix)
    offset_global_bound, offset_global_estimate = get_global_max(offset_error_bound_matrix, offset_error_est_matrix)

    # Results needed: 1) global bound and estimate for angle, offset; 2) global bound and estimate after mask for offset
    # 3) mask stats
    with open(os.path.join(stats_dir, 'statistics.txt'), 'a') as f:
        f.write('Global max results: \n')
        f.write('Global bound for angle is {} \n'.format(angle_global_bound))
        f.write('Global estimate for angle is {} \n'.format(angle_global_estimate))
        f.write('Global bound for offset is {} \n'.format(offset_global_bound))
        f.write('Global estimate for offset is {} \n'.format(offset_global_estimate))
        f.write('\n')
        f.write('Offset mask results: \n')
        f.write('Percentage of trusted region (outside mask) is {} \n'.format(percent_trusted*100))
        f.write('Max estimated error outside the mask is {} \n'.format(max_est_outside))
        f.write('Average estimated error outside the mask is {} \n'.format(avg_est_outside))
        f.write('Max estimated error inside the mask is {} \n'.format(max_est_inside))
        f.write('Average estimated error inside the mask is {} \n'.format(avg_est_inside))
        f.write('Percentage of masked region where the estimated error is larger than the max estimated error outside is {} \n'.format(percent_bad_inside*100))
        f.write('Percentage of masked region where the estimated error is smaller than the average estimated error outside is {} \n'.format(percent_good_inside*100))
        f.write('Max error bound outside the mask is {} \n'.format(max_bound_outside))

def get_percent_below_cutoff(data_array, cutoff):
    return np.count_nonzero(data_array < cutoff) / len(data_array) * 100

def plot_cumulative_histogram(setting, data_matrix, save_dir):
    flattened = np.reshape(data_matrix, -1)
    num_grids = len(flattened)
    max_value = np.max(flattened)
    num_cutoff_points = 100
    cutoffs = np.array(range(num_cutoff_points)) / (num_cutoff_points - 1) * max_value
    percentages = np.zeros(num_cutoff_points)
    for i in tqdm(range(num_cutoff_points)):
        percentages[i] = get_percent_below_cutoff(flattened, cutoffs[i])

    percentile_95 = np.percentile(flattened, 95)
    percentile_99 = np.percentile(flattened, 99)
    plt.figure()
    plt.plot(cutoffs, percentages)
    plt.axvline(x=percentile_95)
    plt.axvline(x=percentile_99)

    plt.xlabel('Threshold value')
    plt.ylabel('Percentage of input space')
    plt.title('Percentage of input space with error below threshold value')
    plot_file_path = os.path.join(save_dir, setting + '_cumulative.png')
    plt.savefig(plot_file_path)


def compute_plot_cumulative_histogram(result_dir, offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix):
    offset_gap_matrix = offset_error_bound_matrix - offset_error_est_matrix
    angle_gap_matrix = angle_error_bound_matrix - angle_error_est_matrix

    stats_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(stats_dir):
        print("Creating {}".format(stats_dir))
        os.makedirs(stats_dir)

    entries = {'offset_lower': offset_error_est_matrix,
                'offset_upper': offset_error_bound_matrix,
                'offset_gap': offset_gap_matrix,
                'angle_lower': angle_error_est_matrix,
                'angle_upper': angle_error_bound_matrix,
                'angle_gap': angle_gap_matrix}
    for setting, matrix in entries.items():
        plot_cumulative_histogram(setting, matrix, stats_dir)


def main():
    parser = argparse.ArgumentParser(description='Get statistics')
    parser.add_argument('--result_dir', help='path to result to get errors')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

    # Read bound and estimate results
    offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix = read_bound_est_result(result_dir)

    # Compute and plot cumulative histogram
    compute_plot_cumulative_histogram(result_dir, offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix)

    # Compute and store stats
    compute_and_store_stats(result_dir, offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
