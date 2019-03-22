import os
import image_generator
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math

# Unit centimeters
scene_params = {
'line_width': 30.0,
'road_width': 300.0, # per lane
'shade_width': 5.0,
}

camera_params = {
'height': 150.0,
'focal_length': 5.0,
'pixel_num': 32,
'pixel_size': 0.8,
}

def generate_dataset(viewer, num_images, offset_range, angle_range):
    # offset_range and angle_range are list, [low, high]
    offsets = np.random.uniform(low=offset_range[0], high=offset_range[1], size=num_images)
    angles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=num_images)
    images = []
    for i in tqdm(range(num_images)):
        images.append(np.expand_dims(viewer.take_picture(offsets[i], angles[i]), axis=0))
    images = np.concatenate(images, axis=0) # (N, H, W)
    dataset = {}
    dataset['images'] = images
    dataset['offsets'] = offsets
    dataset['angles'] = angles
    return dataset

def generate_dataset_for_error_est(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side):
    # offset_range and angle_range are list, [low, high]
    # returned dataset has order: innermost is within a grid, then varying angle, then varying offset
    offsets = []
    angles = []
    images = []
    offset_grid_size = (offset_range[1] - offset_range[0])/offset_grid_num
    angle_grid_size = (angle_range[1] - angle_range[0])/angle_grid_num
    offset_point_space = offset_grid_size / num_points_per_side
    angle_point_space = angle_grid_size / num_points_per_side
    for i in tqdm(range(offset_grid_num)):
        offset_start = offset_range[0] + i * offset_grid_size
        for j in range(angle_grid_num):
            angle_start = angle_range[0] + j * angle_grid_size
            for p in range(num_points_per_side):
                offset = offset_start + p * offset_point_space + offset_point_space/2
                for q in range(num_points_per_side):
                    angle = angle_start + q * angle_point_space + angle_point_space/2
                    images.append(np.expand_dims(viewer.take_picture(offset, angle), axis=0))
                    offsets.append(offset)
                    angles.append(angle)
    dataset = {}
    dataset['images'] = np.concatenate(images, axis=0) # (N, H, W)
    dataset['offsets'] = np.array(offsets) # (N,)
    dataset['angles'] = np.array(angles) # (N,)
    dataset['points_per_grid'] = num_points_per_side * num_points_per_side
    dataset['offset_grid_num'] = offset_grid_num
    dataset['angle_grid_num'] = angle_grid_num
    return dataset

def generate_dataset_for_verify(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num):
    # offset_range and angle_range are list, [low, high]
    offset_grid_size = (offset_range[1] - offset_range[0])/offset_grid_num
    angle_grid_size = (angle_range[1] - angle_range[0])/angle_grid_num
    offset_delta = offset_grid_size / 2
    angle_delta = angle_grid_size / 2
    image_lower_bounds = []
    image_upper_bounds = []
    offset_lower_bounds = []
    offset_upper_bounds = []
    angle_lower_bounds = []
    angle_upper_bounds = []
    images = []
    offsets = []
    angles = []
    for i in tqdm(range(offset_grid_num)):
        offset = offset_range[0] + i * offset_grid_size + offset_delta
        for j in range(angle_grid_num):
            angle = angle_range[0] + j * angle_grid_size + angle_delta
            image, lower_bound_matrix, upper_bound_matrix = viewer.take_picture_with_range(offset, angle, offset_delta, angle_delta)
            image_lower_bounds.append(np.expand_dims(lower_bound_matrix, axis=0))
            image_upper_bounds.append(np.expand_dims(upper_bound_matrix, axis=0))
            images.append(np.expand_dims(image, axis=0))
            offset_lower_bounds.append(offset - offset_delta)
            offset_upper_bounds.append(offset + offset_delta)
            offsets.append(offset)
            angle_lower_bounds.append(angle - angle_delta)
            angle_upper_bounds.append(angle + angle_delta)
            angles.append(angle)

    dataset = {}
    dataset['image_lower_bounds'] = np.concatenate(image_lower_bounds, axis=0) # (N, H, W)
    dataset['image_upper_bounds'] = np.concatenate(image_upper_bounds, axis=0) # (N, H, W)
    dataset['offset_lower_bounds'] = np.array(offset_lower_bounds) # (N,)
    dataset['offset_upper_bounds'] = np.array(offset_upper_bounds) # (N,)
    dataset['angle_lower_bounds'] = np.array(angle_lower_bounds) # (N,)
    dataset['angle_upper_bounds'] = np.array(angle_upper_bounds) # (N,)
    dataset['images'] = np.concatenate(images, axis=0) # (N, H, W)
    dataset['offsets'] = np.array(offsets) # (N,)
    dataset['angles'] = np.array(angles) # (N,)
    dataset['offset_grid_num'] = offset_grid_num
    dataset['angle_grid_num'] = angle_grid_num
    return dataset

def gen_train_valid_data(viewer):
    # Generate training and validation dataset
    offset_range = [-20, 20]
    angle_range = [-20, 20]
    training_size = 10000
    validation_size = 500
    training_set = generate_dataset(viewer, training_size, offset_range, angle_range)
    validation_set = generate_dataset(viewer, validation_size, offset_range, angle_range)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)
    sio.savemat(os.path.join(data_dir, 'train_20_10000.mat'), training_set)
    sio.savemat(os.path.join(data_dir, 'valid_20_500.mat'), validation_set)

def gen_test_data_for_verify(viewer, range, grid_size):
    # Generate a test set for verify
    offset_range = [-range, range]
    angle_range = [-range, range]
    offset_grid_num = int(2*range/grid_size)
    angle_grid_num = int(2*range/grid_size)
    dataset = generate_dataset_for_verify(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num)
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)
    sio.savemat(os.path.join(data_dir, 'test_verify_{}_{}.mat'.format(range, grid_size)), dataset)

def gen_test_data_for_error_est(viewer, range, grid_size):
    # Generate a test set for error estimation
    offset_range = [-range, range]
    angle_range = [-range, range]
    offset_grid_num = int(2*range/grid_size)
    angle_grid_num = int(2*range/grid_size)
    num_points_per_side = math.ceil(grid_size/0.1)
    dataset = generate_dataset_for_error_est(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side)
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)
    sio.savemat(os.path.join(data_dir, 'test_error_est_{}_{}.mat'.format(range, grid_size)), dataset)

def gen_example_picture(viewer):
    # Generate a picture
    offset = -102.0
    angle = 60.0
    image_taken = viewer.take_picture(offset, angle)

    img = Image.fromarray(image_taken)
    img = img.convert("L")
    img.save('test.jpg')

def gen_example_picture_with_range(viewer):
    # Generate a picture with range
    delta_x = 2.0
    delta_phi = 1.0
    image_matrix, lower_bound_matrix, upper_bound_matrix = viewer.take_picture_with_range(offset, angle, delta_x, delta_phi)
    # Don't need the following to generate data, just visualization. There is potentially a alias in converting/displaying as jpeg
    # Store training images as np array (N, H, W) with range [0,255], offsets and angles as (N,)
    # Store pixel ranges as np array with range (0,1.0)
    img = Image.fromarray(image_matrix)
    img = img.convert("L")
    img.save('test.jpg')
    img_low = Image.fromarray(lower_bound_matrix)
    img_low = img_low.convert("L")
    img_low.save('test_low.jpg')
    img_upper = Image.fromarray(upper_bound_matrix)
    img_upper = img_upper.convert("L")
    img_upper.save('test_upper.jpg')

def main():
    scene = image_generator.Scene(scene_params)
    viewer = image_generator.Viewer(camera_params, scene)

    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--range', type=int, help='Range for offset and angle, for generating test datasets')
    parser.add_argument('--grid_size', type=float, help='Grid size for calculating error')
    args = parser.parse_args()

    gen_test_data_for_error_est(viewer, args.range, args.grid_size)
    gen_test_data_for_verify(viewer, args.range, args.grid_size)

    #gen_train_valid_data(viewer)

    #gen_example_picture(viewer)

    #gen_example_picture_with_range(viewer)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
