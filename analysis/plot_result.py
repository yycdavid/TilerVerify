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

def main():
    plot_error_ratio_against_grid_size()

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
