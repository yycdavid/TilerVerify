import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


def plot_error_ratio_against_grid_size():
    # Need to rerun 1.0
    grid_sizes = [1.0, 0.5, 0.25]
    offset_error_ratios = [15.50458213408549, 10.789974015573762, 7.310701659835029]
    angle_error_ratios = [13.50865467776187, 8.569672819560829, 5.845494764120568]
    solve_times = [4077.78815712, 3155.6530118470064, 5158.919815271]

    ticks = [1,2,3]
    plt.plot(ticks, solve_times, color='g', marker='s')
    plt.xticks(ticks, [1.0,0.5,0.25])
    plt.xlabel('grid size')
    #plt.ylabel('error ratio')
    plt.ylabel('Time (s)')
    plt.title('Solve time')
    plt.show()

def plot_global_bound_against_grid_size():
    grid_sizes = [0.1, 0.2, 0.4, 0.8]
    global_bounds = [7.127, 12.034, 22.011, 37.757]
    global_estimate = 4.077
    plt.figure()
    plt.plot(grid_sizes, global_bounds, marker='s', label='bound')
    plt.plot(grid_sizes, [global_estimate for i in global_bounds], marker='s', label='estimate')
    plt.legend(loc='upper left')

    plt.xlabel('grid size')
    plt.ylabel('Global max error')
    plt.title('Global error bound for angle measurement against grid size')
    plot_file_path = 'angle_bound_grid_size.png'
    plt.savefig(plot_file_path)

def plot_trusted_region_against_grid_size():
    grid_sizes = [0.1, 0.2, 0.4, 0.8]
    trusted_percentages = [99.87, 99.08, 90.83, 24.47]
    plt.figure()
    plt.plot(grid_sizes, trusted_percentages, marker='s')

    plt.xlabel('grid size')
    plt.ylabel('Percentage of trusted region')
    plt.title('Percentage of trusted region (5.0) for offset measurement against grid size')
    plot_file_path = 'offset_trusted_grid_size.png'
    plt.savefig(plot_file_path)

def plot_time_against_grid_size():
    grid_sizes = [0.1, 0.2, 0.4, 0.8]
    times = [52607, 32935, 47841, 96731]
    plt.figure()
    plt.plot(grid_sizes, times, marker='s')

    plt.xlabel('grid size')
    plt.ylabel('Time to run verification')
    plt.title('Time for verification against grid size')
    plot_file_path = 'time_grid_size.png'
    plt.savefig(plot_file_path)


def main():
    plot_time_against_grid_size()

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
