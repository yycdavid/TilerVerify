import os
import numpy as np
import argparse
import scipy.io as sio
import h5py


RESULTS_ROOT = 'trained_models'

def get_max_error_in_each_grid(errors, points_per_grid):
    return np.max(np.reshape(errors, (-1, points_per_grid)), axis=1)

def get_bound_to_actual_ratio(bound, actual):
    return np.mean(np.divide(bound, actual))

def main():
    parser = argparse.ArgumentParser(description='Get statistics on error bounds')
    parser.add_argument('--exp_name', help='name of experiment to get statistics')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, args.exp_name)

    # Get error estimates
    error_estimate_file = os.path.join(exp_dir, 'error_est_result.mat')
    error_estimate = sio.loadmat(error_estimate_file)

    points_per_grid = int(error_estimate['points_per_grid'])
    offset_grid_num = int(error_estimate['offset_grid_num'])
    angle_grid_num = int(error_estimate['angle_grid_num'])
    offset_errors_estimate = get_max_error_in_each_grid(error_estimate['offset_errors'], points_per_grid)
    angle_errors_estimate = get_max_error_in_each_grid(error_estimate['angle_errors'], points_per_grid)

    # Get error bounds
    error_bound_file = os.path.join(exp_dir, 'error_bound_result.mat')
    with h5py.File(error_bound_file, 'r') as f:
        error_bound = {}
        for k, v in f.items():
            error_bound[k] = np.array(v)
    offset_errors_bound = error_bound['offset_errors']
    angle_errors_bound = error_bound['angle_errors']

    # Compute statistics
    offset_ratio = get_bound_to_actual_ratio(offset_errors_bound, offset_errors_estimate)
    angle_ratio = get_bound_to_actual_ratio(angle_errors_bound, angle_errors_estimate)

    with open(os.path.join(exp_dir, 'statistics.txt'), 'w') as f:
        f.write('The bound-to-actual ratio for offset is {} \n'.format(offset_ratio))
        f.write('The bound-to-actual ratio for angle is {} \n'.format(angle_ratio))

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
