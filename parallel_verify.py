import os
import argparse
import sys
import generate_data
import multiprocessing as mp

# Use mp pool, Julia, to run julia script for verify in parallel; .jl script will
# operate on a single data file and produce result as a single summary file

# Process and conbine all the summary files to get results, save and clean

def main():
    parser = argparse.ArgumentParser(description='Running verifying experiment with parallelism')
    parser.add_argument('--offset_range', type=int, help='Range for offset, for generating test datasets')
    parser.add_argument('--angle_range', type=int, help='Range for angle, for generating test datasets')
    parser.add_argument('--grid_size', type=float, help='Grid size for calculating error')
    parser.add_argument('--num_threads', type=int, help='Number of threads to run the verifier')
    args = parser.parse_args()

    # Generate data for verify as separate files for each thread, in a subdirectory in data
    generate_data.gen_data_for_verify_parallel(args.offset_range, args.angle_range, args.grid_size, args.num_threads)

    


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
