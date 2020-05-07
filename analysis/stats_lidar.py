import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import h5py
import matplotlib.colors as colors
import csv


def read_csv_count_line(target_file):
    with open(target_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        for row in csv_reader:
            line_count += 1

    return line_count


def get_solved_count(folder, weight):
    count = 0.0
    for label in range(3):
        solved_file = os.path.join(folder, f'{label}_summary.csv')
        if os.path.isfile(solved_file):
            count += read_csv_count_line(solved_file)

    return count * weight


def get_not_solved_count(folder, weight):
    count = 0.0
    for label in range(3):
        solved_file = os.path.join(folder, f'{label}_to_solve.csv')
        if os.path.isfile(solved_file):
            count += read_csv_count_line(solved_file)

    return count * weight


def main_adaptive(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = os.path.join(base_dir, args.result_dir)

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
    verify_file_path = os.path.join(result_dir, 'verify_result.mat')
    if args.hdf:
        with h5py.File(verify_file_path, 'r') as f:
            verify_result = {}
            for k, v in f.items():
                verify_result[k] = np.array(v)
    else:
        verify_result = sio.loadmat(verify_file_path)

    distance_grid_num = int(verify_result['distance_grid_num'])
    angle_grid_num = int(verify_result['angle_grid_num'])

    verified_count = 0
    for label in range(3):
        if args.hdf:
            verify_list = verify_result['VerifyStatus_'+str(label)]
        else:
            verify_list = verify_result['VerifyStatus_'+str(label)][0]
        for status in verify_list:
            if status == 0:
                verified_count += 1

    verified_percentage = verified_count / (distance_grid_num * angle_grid_num * 3) * 100

    with open(os.path.join(result_dir, 'verified_perc.txt'), 'w') as fout:
        fout.write(f'Verified percentage: {verified_percentage}')


def main():
    parser = argparse.ArgumentParser(description='Get stats for lidar')
    parser.add_argument('--result_dir', help='path to result to get heatmap')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--hdf', action='store_true')
    args = parser.parse_args()
    if args.adaptive:
        main_adaptive(args)
    else:
        main_normal(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
