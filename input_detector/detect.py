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


def load_bounding_boxes():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    save_dir = os.path.join(data_dir, 'verify_offset_40_angle_60_grid_0.2_thread_20')

    lower_bounds = []
    upper_bounds = []
    offset_lower_bounds = []
    angle_lower_bounds = []
    for thread_num in tqdm(range(20)):
        sub_dataset = sio.loadmat(os.path.join(save_dir, 'thread_{}.mat'.format(thread_num)))
        lower_bounds.append(sub_dataset['image_lower_bounds'])
        upper_bounds.append(sub_dataset['image_upper_bounds'])
        offset_lower_bounds.append(sub_dataset['offset_lower_bounds'].squeeze())
        angle_lower_bounds.append(sub_dataset['angle_lower_bounds'].squeeze())
    lower_bounds = np.concatenate(lower_bounds, axis=0)
    upper_bounds = np.concatenate(upper_bounds, axis=0)
    offset_lower_bounds = np.concatenate(offset_lower_bounds)
    angle_lower_bounds = np.concatenate(angle_lower_bounds)
    N = lower_bounds.shape[0]
    lower_bounds = np.reshape(lower_bounds, (N, -1))
    upper_bounds = np.reshape(upper_bounds, (N, -1))
    return lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds


def get_bounding_boxes(file_name, mode):
    # Check if it exists. If not, create one
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    file_path = os.path.join(data_dir, file_name)
    if os.path.isfile(file_path):
        print("Loading bounding boxes...")
        with open(file_path, 'rb') as f:
            bounding_boxes = pickle.load(f)
        with open(os.path.join(data_dir, 'ground_truth_bounds.mat'), 'rb') as f:
            ground_truth_bounds = pickle.load(f)
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
        lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds = load_bounding_boxes()
        bounding_boxes['lower_bounds'] = lower_bounds
        bounding_boxes['upper_bounds'] = upper_bounds

        with open(file_path, 'wb') as f:
            pickle.dump(bounding_boxes, f, protocol=pickle.HIGHEST_PROTOCOL)

        ground_truth_bounds = {}
        ground_truth_bounds['offset_lower_bounds'] = offset_lower_bounds
        ground_truth_bounds['angle_lower_bounds'] = angle_lower_bounds
        with open(os.path.join(data_dir, 'ground_truth_bounds.mat'), 'wb') as f:
            pickle.dump(ground_truth_bounds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bounding_boxes['lower_bounds'], bounding_boxes['upper_bounds'], ground_truth_bounds['offset_lower_bounds'], ground_truth_bounds['angle_lower_bounds']


def get_input_image():
    viewer = get_viewer()
    offset = np.random.uniform(low=-40.0, high=40.0)
    angle = np.random.uniform(low=-60.0, high=60.0)
    print("Example image offset: {}, angle: {}".format(offset, angle))
    example_image = viewer.take_picture(offset, angle)
    return example_image.flatten(), offset, angle


def main():
    parser = argparse.ArgumentParser(description='Detecting whether input is legal or not')
    parser.add_argument('--file_name', type=str, help='File path to bounding boxes')
    parser.add_argument('--mode', type=str, help='can be naive')
    args = parser.parse_args()
    # Load bounding boxes
    if args.mode == 'naive':
        lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds = get_bounding_boxes(args.file_name, 'naive')
    # Get an input image
    print('Get example images...')
    example_images = []
    num_images = 20
    for i in range(num_images):
        example_image, _, _ = get_input_image()
        example_images.append(np.expand_dims(example_image, axis=0))

    # Decide if it's in any of the bounding box
    print("Start detecting...")
    start_t = time.time()
    for i in range(num_images):
        example_image = example_images[i]
        is_legal = np.any(np.logical_and(np.all(lower_bounds <= example_image, axis=1), np.all(example_image <= upper_bounds, axis=1)))
        print(is_legal)

    end_t = time.time()
    print('Time spent per input: {}'.format((end_t - start_t)/num_images))




if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
