import os
import numpy as np
import argparse
import scipy.io as sio
import h5py
import csv


RESULTS_ROOT = 'trained_models'

def get_total_time_for_verify(model_name, exp_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, model_name, exp_name)
    summary_file_path = os.path.join(exp_dir, 'summary.csv')
    total_time = 0.0
    with open(summary_file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_time += float(row["SolveTime"])

    stats_file_path = os.path.join(exp_dir, 'statistics.txt')
    with open(stats_file_path, 'a') as f:
        f.write('Total solve time is {} seconds.\n'.format(total_time))

def main():
    model_name = '20_10000'
    exp_names = ['offset_20_angle_20_grid_size_1.0']
    for exp_name in exp_names:
        get_total_time_for_verify(model_name, exp_name)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
