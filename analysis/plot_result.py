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
    plot_file_path = 'angle_bound_grid_size.pdf'
    plt.savefig(plot_file_path)

def plot_99_against_grid_size():
    grid_sizes = [0.05, 0.1, 0.2, 0.4, 0.8]
    bounds_angle = [2.98, 3.69, 5.26, 9.08, 21.33]
    bounds_offset = [2.05, 2.65, 4.84, 12.46, 33.34]
    gap_angle = [1.01, 1.9, 3.71, 7.66, 19.66]
    gap_offset = [0.69, 1.41, 3.64, 11.4, 32.46]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    color = 'tab:red'
    ax1.set_xlabel('Grid size')
    ax1.set_ylabel('Angle (degree)', color=color)
    lns1 = ax1.plot(grid_sizes, bounds_angle, marker='s', color=color, label='angle bound')
    lns2 = ax1.plot(grid_sizes, gap_angle, marker='^', color=color, label='angle gap')
    #ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Offset (length unit)', color=color)
    lns3 = ax2.plot(grid_sizes, bounds_offset, marker='s', color=color, label='offset bound')
    lns4 = ax2.plot(grid_sizes, gap_offset, marker='^', color=color, label='offset gap')

    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title('99 percentiles of bounds and gaps\n against grid size')
    plot_file_path = '99_percentile_grid_size.pdf'
    plt.savefig(plot_file_path)

def plot_trusted_region_against_grid_size():
    grid_sizes = [0.05, 0.1, 0.2, 0.4, 0.8]
    trusted_percentages_offset = [99.50, 98.50, 90.56, 35.29, 0.06]
    trusted_percentages_angle = [99.83, 98.75, 82.11, 8.455, 0.0]
    plt.figure()
    plt.plot(grid_sizes, trusted_percentages_offset, marker='s', label='offset')
    plt.plot(grid_sizes, trusted_percentages_angle, marker='s', label='angle')
    plt.legend(loc='upper right')

    plt.xlabel('grid size')
    plt.ylabel('Percentage of trusted region')
    plt.title('Percentage of trusted region\n against grid size')
    plot_file_path = 'trusted_grid_size.pdf'
    plt.savefig(plot_file_path)

def plot_time_against_grid_size():
    grid_sizes = [0.05, 0.1, 0.2, 0.4, 0.8]
    times = [253788, 52607, 32935, 47841, 96731]
    times = [time/3600 for time in times]
    plt.figure()
    plt.plot(grid_sizes, times, marker='s')
    plt.xlabel('grid size')
    plt.ylabel('Time for solving (hour)')
    plt.title('Time for solving against grid size')

    plot_file_path = 'time_grid_size.pdf'
    plt.savefig(plot_file_path)


def plot_tradeoff():
    fixed_time = [253788, 52607, 32935, 47841, 96731]
    fixed_time = [time/3600 for time in fixed_time]
    fixed_perc = [99.50, 98.50, 90.56, 35.29, 0.06]
    adaptive_time = 32593/3600
    # 23995 + 6680 + 1918
    adaptive_perc = 99.70

    plt.figure()
    plt.plot(fixed_time, fixed_perc, marker='s', label='fixed')
    plt.xlabel('Verification time (hours)')
    plt.ylabel('Percentage verified')

    plt.scatter(adaptive_time, adaptive_perc, marker='x', c='r', label='adaptive')
    plt.legend(loc='lower right')

    plot_file_path = 'trade_off_1.pdf'
    plt.savefig(plot_file_path)


def plot_tradeoff_perc():
    fixed_time = [253788, 52607, 32935, 47841, 96731]
    fixed_time = [time/3600 for time in fixed_time]
    fixed_perc = [99.50, 98.50, 90.56, 35.29, 0.06]
    grid_sizes = [0.05, 0.1, 0.2, 0.4, 0.8]
    adaptive_time = 32593/3600
    # 23995 + 6680 + 1918
    adaptive_perc = 99.70

    plt.figure()
    plt.plot(grid_sizes, fixed_perc, marker='s', label='fixed')
    plt.xlabel('Grid size')
    plt.ylabel('Percentage verified')

    plt.plot(grid_sizes, [adaptive_perc for _ in grid_sizes], c='r', label='adaptive')
    plt.legend(loc='upper right')

    plot_file_path = 'trade_off_perc.pdf'
    plt.savefig(plot_file_path)

def plot_tradeoff_time():
    fixed_time = [253788, 52607, 32935, 47841, 96731]
    fixed_time = [time/3600 for time in fixed_time]
    fixed_perc = [99.50, 98.50, 90.56, 35.29, 0.06]
    grid_sizes = [0.05, 0.1, 0.2, 0.4, 0.8]
    adaptive_time = 32593/3600
    # 23995 + 6680 + 1918
    adaptive_perc = 99.70

    plt.figure()
    plt.plot(grid_sizes, fixed_time, marker='s', label='fixed')
    plt.xlabel('Grid size')
    plt.ylabel('Verification time (hours)')

    plt.plot(grid_sizes, [adaptive_time for _ in grid_sizes], c='r', label='adaptive')
    plt.legend(loc='upper right')

    plot_file_path = 'trade_off_time.pdf'
    plt.savefig(plot_file_path)


def main():
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.autolayout': True})
    #plot_99_against_grid_size()
    #plot_trusted_region_against_grid_size()
    #plot_time_against_grid_size()
    plot_tradeoff_perc()
    plot_tradeoff_time()

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
