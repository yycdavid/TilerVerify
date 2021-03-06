import os
import numpy as np
import argparse
import scipy.io as sio
import h5py
from heatmap import get_error_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

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
    if percent_trusted == 0:
        return percent_trusted, 0, 0, 0, 0, 0, 0, 0
    else:
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

    cap_percentage = 0.03
    cap_value_offset = cap_percentage * 80
    cap_value_angle = cap_percentage * 120
    percent_trusted_offset, max_est_outside, avg_est_outside, max_est_inside, avg_est_inside, percent_bad_inside, percent_good_inside, max_bound_outside = get_stats_about_masked_region(offset_error_bound_matrix, offset_error_est_matrix, cap_value_offset)

    percent_trusted_angle, _, _, _, _, _, _, _ = get_stats_about_masked_region(angle_error_bound_matrix, angle_error_est_matrix, cap_value_angle)

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
        f.write('Percentage of trusted region for offset (outside mask) is {} \n'.format(percent_trusted_offset*100))
        f.write('Percentage of trusted region for angle (outside mask) is {} \n'.format(percent_trusted_angle*100))
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
    plt.axvline(x=percentile_95, color='r', label="95%: {:.2f}".format(percentile_95))
    plt.axvline(x=percentile_99, color='m', label="99%: {:.2f}".format(percentile_99))
    plt.legend(loc='lower right')

    if setting.split('_')[0] == 'offset':
        plt.xlabel('Threshold (length unit)')
    elif setting.split('_')[0] == 'angle':
        plt.xlabel('Threshold (degree)')
    plt.ylabel('% state space')
    if setting == 'offset_upper':
        plt.title('offset error bound')
    elif setting == 'offset_gap':
        plt.title('offset error gap')
    elif setting == 'angle_upper':
        plt.title('angle error bound')
    elif setting == 'angle_gap':
        plt.title('angle error gap')
    plot_file_path = os.path.join(save_dir, setting + '_cumulative.pdf')
    plt.savefig(plot_file_path, bbox_inches='tight')


def compute_plot_cumulative_histogram(result_dir, offset_error_est_matrix, angle_error_est_matrix, offset_error_bound_matrix, angle_error_bound_matrix):
    offset_gap_matrix = offset_error_bound_matrix - offset_error_est_matrix
    angle_gap_matrix = angle_error_bound_matrix - angle_error_est_matrix

    stats_dir = os.path.join(result_dir, "plots_and_stats")
    if not os.path.exists(stats_dir):
        print("Creating {}".format(stats_dir))
        os.makedirs(stats_dir)

    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.autolayout': True})

    entries = {'offset_lower': offset_error_est_matrix,
                'offset_upper': offset_error_bound_matrix,
                'offset_gap': offset_gap_matrix,
                'angle_lower': angle_error_est_matrix,
                'angle_upper': angle_error_bound_matrix,
                'angle_gap': angle_gap_matrix}
    for setting, matrix in entries.items():
        plot_cumulative_histogram(setting, matrix, stats_dir)



def read_csv_count_line(target_file):
    with open(target_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        for row in csv_reader:
            line_count += 1

    return line_count


def get_solved_count(folder, weight):
    count = 0.0
    solved_file = os.path.join(folder, 'summary.csv')
    if os.path.isfile(solved_file):
        count += read_csv_count_line(solved_file)

    return count * weight


def get_not_solved_count(folder, weight):
    count = 0.0
    solved_file = os.path.join(folder, 'to_solve.csv')
    if os.path.isfile(solved_file):
        count += read_csv_count_line(solved_file)

    return count * weight


def main_adaptive(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

    if args.plot_final_size > 0:
        current_size = args.plot_final_size
        grid_sizes = []
        solved_counts = []

        weight = 1.0

        grid_sizes.append(current_size)
        solved_counts.append(get_solved_count(result_dir, weight))

        next_level = '1'
        if args.limit_max_level:
            while int(next_level) <= 2:
                next_dir = os.path.join(result_dir, next_level)
                weight = weight / 4
                current_size = current_size / 2
                grid_sizes.append(current_size)
                solved_counts.append(get_solved_count(next_dir, weight))

                next_level = str(int(next_level)+1)

        else:
            while os.path.isdir(os.path.join(result_dir, next_level)):
                next_dir = os.path.join(result_dir, next_level)
                weight = weight / 4
                current_size = current_size / 2
                grid_sizes.append(current_size)
                solved_counts.append(get_solved_count(next_dir, weight))

                next_level = str(int(next_level)+1)

        last_dir = next_dir
        not_solved_count = get_not_solved_count(last_dir, weight)

        solved_counts[-1] += not_solved_count

        total_counts = sum(solved_counts)
        solved_perc = [c/total_counts*100 for c in solved_counts]

        plt.figure()
        plt.plot(grid_sizes, solved_perc, marker='s')
        plt.xlabel('Grid size')
        plt.ylabel('Percentage of regions with final tile size')

        plot_file_path = os.path.join(result_dir, 'final_size_dist.pdf')
        plt.savefig(plot_file_path)

    else:

        solved_count = 0.0

        weight = 1.0

        solved_count += get_solved_count(result_dir, weight)

        next_level = '1'
        while os.path.isdir(os.path.join(result_dir, next_level)):
            next_dir = os.path.join(result_dir, next_level)
            weight = weight / 4
            solved_count += get_solved_count(next_dir, weight)

            next_level = str(int(next_level)+1)

        last_dir = next_dir
        not_solved_count = get_not_solved_count(last_dir, weight)

        verified_percentage = solved_count / (solved_count + not_solved_count) * 100

        with open(os.path.join(result_dir, 'verified_perc.txt'), 'w') as fout:
            fout.write(f'Verified percentage: {verified_percentage}')


def main_normal(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

    # Plot error bound maps
    verify_file_path = os.path.join(result_dir, 'error_bound_result.mat')

    error_bound_file_path = os.path.join(result_dir, 'error_bound_result.mat')
    with h5py.File(error_bound_file_path, 'r') as f:
        error_bound_result = {}
        for k, v in f.items():
            error_bound_result[k] = np.array(v)

    distance_grid_num = int(error_bound_result['offset_grid_num'])
    angle_grid_num = int(error_bound_result['angle_grid_num'])

    verified_count = 0

    for i in range(distance_grid_num * angle_grid_num):
        if (error_bound_result['angle_errors'][i] < args.angle_err_thresh) and (error_bound_result['offset_errors'][i] < args.offset_err_thresh):
            verified_count += 1

    verified_percentage = verified_count / (distance_grid_num * angle_grid_num) * 100

    with open(os.path.join(result_dir, 'verified_perc.txt'), 'w') as fout:
        fout.write(f'Verified percentage (offset error < {args.offset_err_thresh}, angle error < {args.angle_err_thresh}): {verified_percentage}')


def main_perc_verified():
    parser = argparse.ArgumentParser(description='Get stats for lidar')
    parser.add_argument('--result_dir', help='path to result to get heatmap')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--limit_max_level', action='store_true')
    parser.add_argument('--offset_err_thresh', type=float, default=0.0)
    parser.add_argument('--angle_err_thresh', type=float, default=0.0)
    parser.add_argument('--plot_final_size', type=float, default=-0.1)
    args = parser.parse_args()

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.autolayout': True})
    
    if args.adaptive:
        main_adaptive(args)
    else:
        main_normal(args)


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
        main_perc_verified()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
