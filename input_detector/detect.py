import os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
import time

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not base_dir in sys.path:
    sys.path.append(base_dir)
from generate_data import get_viewer


def gen_bounding_boxes(viewer, offset_rng, angle_rng, grid_size):
    # Generate a test set for verify
    offset_range = [-offset_rng, offset_rng]
    angle_range = [-angle_rng, angle_rng]
    offset_grid_num = int(2*offset_rng/grid_size)
    angle_grid_num = int(2*angle_rng/grid_size)

    # offset_range and angle_range are list, [low, high]
    offset_grid_size = grid_size
    angle_grid_size = grid_size
    offset_delta = offset_grid_size / 2
    angle_delta = angle_grid_size / 2
    image_lower_bounds = []
    image_upper_bounds = []
    for i in tqdm(range(offset_grid_num)):
        offset = offset_range[0] + i * offset_grid_size + offset_delta
        for j in range(angle_grid_num):
            angle = angle_range[0] + j * angle_grid_size + angle_delta
            _, lower_bound_matrix, upper_bound_matrix = viewer.take_picture_with_range(offset, angle, offset_delta, angle_delta)
            # Reshape to suitable shape
            image_lower_bounds.append(np.expand_dims(lower_bound_matrix.flatten(), axis=0))
            image_upper_bounds.append(np.expand_dims(upper_bound_matrix.flatten(), axis=0))

    return np.concatenate(image_lower_bounds, axis=0), np.concatenate(image_upper_bounds, axis=0)


def load_bounding_boxes()

def get_bounding_boxes(file_name, mode):
    # Check if it exists. If not, create one
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    file_path = os.path.join(data_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            bounding_boxes = pickle.load(f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Creating bounding boxes...")
        bounding_boxes = {}
        '''
        viewer = get_viewer()
        offset_rng = 40
        angle_rng = 60
        grid_size = 0.2
        lower_bounds, upper_bounds = gen_bounding_boxes(viewer, offset_rng, angle_rng, grid_size)
        '''
        lower_bounds, upper_bounds = load_bounding_boxes()
        bounding_boxes['lower_bounds'] = lower_bounds
        bounding_boxes['upper_bounds'] = upper_bounds

        with open(file_path, 'wb') as f:
            pickle.dump(bounding_boxes, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bounding_boxes['lower_bounds'], bounding_boxes['upper_bounds']


def get_input_image():
    viewer = get_viewer()
    offset = np.random.uniform(low=-40.0, high=40.0)
    angle = np.random.uniform(low=-60.0, high=60.0)
    example_image = viewer.take_picture(offset, angle)
    return example_image.flatten()


def main():
    parser = argparse.ArgumentParser(description='Detecting whether input is legal or not')
    parser.add_argument('--file_name', type=str, help='File path to bounding boxes')
    parser.add_argument('--mode', type=str, help='can be naive')
    args = parser.parse_args()
    # Load bounding boxes
    if args.mode == 'naive':
        lower_bounds, upper_bounds = get_bounding_boxes(args.file_name, 'naive')
    # Get an input image
    example_image = np.expand_dims(get_input_image(), axis=0)

    # Decide if it's in any of the bounding box
    start_t = time.time()
    is_legal = np.any(np.logical_and(np.all(lower_bounds <= example_image, axis=1), np.all(example_image <= upper_bounds, axis=1)))
    end_t = time.time()
    print(is_legal)
    print('Time spent: {}'.format(end_t - start_t))


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
