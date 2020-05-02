import numpy as np
import os
import json
import scipy.io as sio
import argparse
import csv

VERIFIED_TRUE = 0
VERIFIED_FALSE = 1
NOT_SURE = 2


def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--exp_dir', type=str, default='test', help='folder storing results')
    parser.add_argument('--save-folder', type=str, default='', help='folder that save verification results')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads')
    parser.add_argument('--adaptive', action='store_true')
    return parser.parse_args()


def convert_solved_status(status):
    # status: str
    if status == 'UNSAT':
        return VERIFIED_TRUE
    elif status == 'SAT':
        return VERIFIED_FALSE
    else:
        return NOT_SURE


def main():
    args = get_args()
    if args.adaptive:
        main_adaptive(args)
    else:
        main_normal(args)


def main_adaptive(args):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(base_dir, 'data', args.exp_dir)
    data_dir = os.path.join(exp_dir, args.save_folder)

    # Read in: verify status, distanceMin, distanceMax, AngleMin, AngleMax, store in CSV
    time = {}
    total_solve_times = []
    total_build_times = []
    total_times = []

    fieldnames = ['SampleNumber', 'DistanceMin', 'DistanceMax', 'AngleMin', 'AngleMax', 'VerifyStatus']

    for label in range(3):
        solved_csv = []
        to_solve_csv = []

        for thread in range(args.num_threads):
            result_file = os.path.join(data_dir, f'{label}_summary_{thread}.json')

            if os.path.isfile(result_file):
                time_for_thread = {}
                time_for_thread['total_build_time'] = 0
                time_for_thread['total_solve_time'] = 0

                with open(result_file, 'r') as f:
                    result_dict = json.load(f)

                for index, results in result_dict.items():
                    csv_row = {}
                    csv_row['SampleNumber'] = index
                    csv_row['DistanceMin'] = results['distance_min']
                    csv_row['DistanceMax'] = results['distance_max']
                    csv_row['AngleMin'] = results['angle_min']
                    csv_row['AngleMax'] = results['angle_max']
                    csv_row['VerifyStatus'] = results['result']

                    if results['result'] == "UNSAT":
                        solved_csv.append(csv_row)
                    else:
                        to_solve_csv.append(csv_row)

                    time_for_thread['total_build_time'] += results['build_time']
                    time_for_thread['total_solve_time'] += results['solve_time']

                time_for_thread['total_time'] = time_for_thread['total_build_time'] + time_for_thread['total_solve_time']
                time[f'{label}_{thread}'] = time_for_thread

                total_solve_times.append(time_for_thread['total_solve_time'])
                total_build_times.append(time_for_thread['total_build_time'])
                total_times.append(time_for_thread['total_time'])

        # Save csv
        if len(solved_csv) > 0:
            save_file = os.path.join(data_dir, f'{label}_summary.csv')
            with open(save_file, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for csv_row in solved_csv:
                    writer.writerow(csv_row)

        if len(to_solve_csv) > 0:
            save_file = os.path.join(data_dir, f'{label}_to_solve.csv')
            with open(save_file, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for csv_row in to_solve_csv:
                    writer.writerow(csv_row)


    time['max_total_time'] = max(total_times)
    time['max_solve_time'] = max(total_solve_times)
    time['max_build_time'] = max(total_build_times)

    with open(os.path.join(data_dir, 'time.json'), 'w') as f:
        json.dump(time, f, indent=2)


def main_normal(args):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(base_dir, 'data', args.exp_dir)
    data_dir = os.path.join(exp_dir, args.save_folder)

    verify_result = {}
    time = {}
    total_solve_times = []
    total_build_times = []
    total_times = []

    info = sio.loadmat(os.path.join(exp_dir, 'info.mat'))
    distance_grid_num = info['distance_grid_num'].item()
    angle_grid_num = info['angle_grid_num'].item()
    num_tiles_per_class = distance_grid_num * angle_grid_num
    verify_result['distance_grid_num'] = distance_grid_num
    verify_result['angle_grid_num'] = angle_grid_num

    for label in range(3):
        verify_results = [-1 for i in range(num_tiles_per_class)]

        for thread in range(args.num_threads):
            time_for_thread = {}
            time_for_thread['total_build_time'] = 0
            time_for_thread['total_solve_time'] = 0

            with open(os.path.join(data_dir, f'{label}_summary_{thread}.json'), 'r') as f:
                result_dict = json.load(f)

            for index, results in result_dict.items():
                verify_results[int(index)] = convert_solved_status(results['result'])
                time_for_thread['total_build_time'] += results['build_time']
                time_for_thread['total_solve_time'] += results['solve_time']

            time_for_thread['total_time'] = time_for_thread['total_build_time'] + time_for_thread['total_solve_time']
            time[f'{label}_{thread}'] = time_for_thread

            total_solve_times.append(time_for_thread['total_solve_time'])
            total_build_times.append(time_for_thread['total_build_time'])
            total_times.append(time_for_thread['total_time'])

        verify_result[f'VerifyStatus_{label}'] = verify_results

    time['max_total_time'] = max(total_times)
    time['max_solve_time'] = max(total_solve_times)
    time['max_build_time'] = max(total_build_times)

    with open(os.path.join(data_dir, 'time.json'), 'w') as f:
        json.dump(time, f, indent=2)

    sio.savemat(os.path.join(data_dir, 'verify_result.mat'), verify_result)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
