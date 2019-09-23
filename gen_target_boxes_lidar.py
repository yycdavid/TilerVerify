import os
import lidar_generator
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
from enum import Enum
import h5py

class Shape(Enum):
    RECTANGLE = 0
    TRIANGLE = 1
    CIRCLE = 2

scene_params_lidar = {
'stick_width': 1.0,
'stick_height': 40.0,
}

sensor_params_lidar = {
'height': 40.0,
'ray_num': 32,
'focal_length': 4.0,
'pixel_size': 0.1,
'max_distance': 300.0,
}


def get_sensor(shape, noise_mode='none', noise_scale=0.0):
    if shape == Shape.RECTANGLE:
        scene_params_lidar['shape'] = 'rectangle'
        scene_params_lidar['side_length'] = 10.0
    elif shape == Shape.TRIANGLE:
        scene_params_lidar['shape'] = 'triangle'
        scene_params_lidar['side_length'] = 10.0
    else:
        scene_params_lidar['shape'] = 'circle'
        scene_params_lidar['radius'] = 5.0
    scene = lidar_generator.Scene(scene_params_lidar)
    sensor = lidar_generator.Sensor(sensor_params_lidar, scene, noise_mode, noise_scale)
    return sensor


def gen_target_boxes(labels, distance_lower_bounds, distance_upper_bounds, angle_lower_bounds, angle_upper_bounds):
    image_lower_bounds = []
    image_upper_bounds = []
    N = len(labels)

    for i in range(N):
        sensor = get_sensor(labels[i], noise_mode='gaussian', noise_scale=0.001)
        distance = (distance_lower_bounds[i] + distance_upper_bounds[i])/2
        distance_delta = (distance_upper_bounds[i] - distance_lower_bounds[i])/2
        angle = (angle_lower_bounds[i] + angle_upper_bounds[i])/2
        angle_delta = (angle_upper_bounds[i] - angle_lower_bounds[i])/2
        lower_bound_matrix, upper_bound_matrix = sensor.take_measurement_with_range(distance, angle, distance_delta, angle_delta)
        image_lower_bounds.append(np.expand_dims(lower_bound_matrix, axis=0))
        image_upper_bounds.append(np.expand_dims(upper_bound_matrix, axis=0))

    results = {}
    results['image_lower_bounds'] = np.concatenate(image_lower_bounds, axis=0) # (N, H, W)
    results['image_upper_bounds'] = np.concatenate(image_upper_bounds, axis=0) # (N, H, W)
    results['distance_lower_bounds'] = np.array(distance_lower_bounds) # (N,)
    results['distance_upper_bounds'] = np.array(distance_upper_bounds) # (N,)
    results['angle_lower_bounds'] = np.array(angle_lower_bounds) # (N,)
    results['angle_upper_bounds'] = np.array(angle_upper_bounds) # (N,)
    results['labels'] = np.array(labels)
    return results


def main():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    exp_dir = os.path.join(data_dir, 'lidar_distance_min_30_max_60_angle_45_grid_0.5_thread_21gaussian0.001_small')

    verify_file_path = os.path.join(exp_dir, 'verify_result.mat')
    with h5py.File(verify_file_path, 'r') as f:
        verify_result = {}
        for k, v in f.items():
            verify_result[k] = np.array(v)

    distance_grid_num = int(verify_result['distance_grid_num'])
    angle_grid_num = int(verify_result['angle_grid_num'])
    distance_min = 30
    distance_max = 60
    angle_range = 45
    distance_range = [distance_min, distance_max]
    angle_range = [-angle_range, angle_range]

    # Compute distance bounds
    d_inv_start = 1.0 / distance_range[1]
    d_inv_end = 1.0 / distance_range[0]
    d_inv_step = (d_inv_end - d_inv_start) / distance_grid_num
    d_inv_bounds = np.arange(d_inv_start, d_inv_end + 1e-5, d_inv_step)
    d_bounds = np.flip(1.0/d_inv_bounds)
    assert len(d_bounds) == distance_grid_num + 1, "distance bounds count incorrect"

    # Angle bounds
    a_step = (angle_range[1] - angle_range[0]) / angle_grid_num
    a_bounds = np.arange(angle_range[0], angle_range[1]+1e-5, a_step)
    assert len(a_bounds) == angle_grid_num + 1, "angle bounds count incorrect"

    # Specify boxes to generate
    labels = []
    distance_lower_bounds = []
    distance_upper_bounds = []
    angle_lower_bounds = []
    angle_upper_bounds = []
    verify_matrix_0 = np.reshape(verify_result['VerifyStatus_0'], (angle_grid_num, distance_grid_num))
    verify_matrix_2 = np.reshape(verify_result['VerifyStatus_2'], (angle_grid_num, distance_grid_num))

    indices_0 = [(90,6), (170,8), (0,34), (90,77), (90,113), (0,76), (179,104), (50,74)]
    indices_2 = [(2,77), (7,82), (66,112), (90,112), (113,113), (177,77)]

    for (a_id, d_id) in indices_0:
        assert verify_matrix_0[a_id, d_id] == 1, 'Specified index is not verified false'
        labels.append(0)
        distance_lower_bounds.append(d_bounds[d_id])
        distance_upper_bounds.append(d_bounds[d_id+1])
        angle_lower_bounds.append(a_bounds[a_id])
        angle_upper_bounds.append(a_bounds[a_id+1])

    for (a_id, d_id) in indices_2:
        assert verify_matrix_2[a_id, d_id] == 1, 'Specified index is not verified false'
        labels.append(2)
        distance_lower_bounds.append(d_bounds[d_id])
        distance_upper_bounds.append(d_bounds[d_id+1])
        angle_lower_bounds.append(a_bounds[a_id])
        angle_upper_bounds.append(a_bounds[a_id+1])

    # Get bounding boxes
    results = gen_target_boxes(labels, distance_lower_bounds, distance_upper_bounds, angle_lower_bounds, angle_upper_bounds)

    save_dir = os.path.join(exp_dir, 'adv_inputs')
    if not os.path.exists(save_dir):
        print("Creating {}".format(save_dir))
        os.makedirs(save_dir)
    sio.savemat(os.path.join(save_dir, 'target_boxes.mat'), results)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
