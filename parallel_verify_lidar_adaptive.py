import os
import argparse
import sys
import generate_data_lidar
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(description='Running verifying experiment with parallelism')
    parser.add_argument('--num_threads', type=int, help='Number of threads to run the verifier')
    parser.add_argument('--noise', type=str, default='none', help='Noise mode, can be none/uniform/gaussian')
    parser.add_argument('--noise_scale', type=float, default=0.05, help='Scale of noise, for uniform it is the max, for gaussian it is one sigma')
    parser.add_argument('--read_from_folder', type=str, default='', help='folder to read further divide boxes')
    parser.add_argument('--write_to_folder', type=str, default='', help='folder to write further divide boxes')
    parser.add_argument('--angle_min_size', type=float, default=0.05, help='path to file to further divide boxes')
    args = parser.parse_args()


    generate_data_lidar.gen_bbox_from_file(args.read_from_folder, args.write_to_folder, args.num_threads, args.noise, args.noise_scale, args.angle_min_size)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
