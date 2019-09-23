import os
import lidar_generator
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
import multiprocessing as mp
from enum import Enum


class Shape(Enum):
    RECTANGLE = 0
    TRIANGLE = 1
    CIRCLE = 2

# For Lidar, 1 Unit = 5 centimeters
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


def generate_dataset(sensor, num_images, distance_range, angle_range):
    # distance_range and angle_range are list, [low, high]
    distances = np.random.uniform(low=distance_range[0], high=distance_range[1], size=num_images)
    angles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=num_images)
    images = []
    for i in tqdm(range(num_images)):
        images.append(np.expand_dims(sensor.take_measurement(angles[i], distances[i]), axis=0))
    images = np.concatenate(images, axis=0) # (N, H, W)
    dataset = {}
    dataset['images'] = images
    return dataset


def generate_partial_train_dataset(shape, noise_mode, noise_scale, num_images, distance_range, angle_range):
    sensor = get_sensor(shape, noise_mode, noise_scale)
    # distance_range and angle_range are list, [low, high]
    distances = np.random.uniform(low=distance_range[0], high=distance_range[1], size=num_images)
    angles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=num_images)
    images = []
    for i in tqdm(range(num_images)):
        images.append(np.expand_dims(sensor.take_measurement(angles[i], distances[i]), axis=0))
    images = np.concatenate(images, axis=0) # (N, H, W)
    dataset = {}
    dataset['images'] = images
    return dataset


def gen_train_valid_data(distance_min_train, distance_max_train, angle_range_train, distance_min_valid, distance_max_valid, angle_range_valid, noise_mode, noise_scale, target_dir_name):
    num_threads = 20
    # Generate training data
    training_size_per_class = 50000
    training_size_per_thread = int(training_size_per_class/num_threads)
    distance_range_train = [distance_min_train, distance_max_train]
    angle_range_train = [-angle_range_train, angle_range_train]
    training_data = {}
    for shape in Shape:
        pool = mp.Pool(processes=num_threads)
        results = [pool.apply_async(generate_partial_train_dataset, args=(shape, noise_mode, noise_scale*5, training_size_per_thread, distance_range_train, angle_range_train)) for _ in range(num_threads)]
        sub_datasets = []
        for i in range(num_threads):
            sub_datasets.append(results[i].get())
        training_data[shape] = {}
        training_data[shape]['images'] = np.concatenate([sub_datasets[i]['images'] for i in range(num_threads)], axis=0)
        training_data[shape]['labels'] = np.array([shape.value for _ in range(training_size_per_class)])

    training_set = {}
    training_set['images'] = np.concatenate([training_data[shape]['images'] for shape in Shape], axis=0)
    training_set['labels'] = np.concatenate([training_data[shape]['labels'] for shape in Shape])

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', target_dir_name)
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, 'train.pickle'), 'wb') as f:
        pickle.dump(training_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate validation data
    validation_size_per_class = 500
    distance_range_valid = [distance_min_valid, distance_max_valid]
    angle_range_valid = [-angle_range_valid, angle_range_valid]
    validation_data = {}
    for shape in Shape:
        sensor = get_sensor(shape, noise_mode, noise_scale)
        validation_data[shape] = generate_dataset(sensor, validation_size_per_class, distance_range_valid, angle_range_valid)
        validation_data[shape]['labels'] = np.array([shape.value for _ in range(validation_size_per_class)])

    validation_set = {}
    validation_set['images'] = np.concatenate([validation_data[shape]['images'] for shape in Shape], axis=0)
    validation_set['labels'] = np.concatenate([validation_data[shape]['labels'] for shape in Shape])

    with open(os.path.join(data_dir, 'valid.pickle'), 'wb') as f:
        pickle.dump(validation_set, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_example_picture(sensor):
    # Generate a picture
    distance = 10.0
    angle = 15.0
    image_taken = sensor.take_picture(distance, angle)

    img = Image.fromarray(image_taken)
    img = img.convert("L")
    img.save('32_distance_{}_angle_{}.pdf'.format(distance, angle))


def gen_example_picture_with_range(sensor):
    # Generate a picture with range
    delta_x = 2.0
    delta_phi = 1.0
    image_matrix, lower_bound_matrix, upper_bound_matrix = sensor.take_picture_with_range(distance, angle, delta_x, delta_phi)
    # Don't need the following to generate data, just visualization. There is potentially a alias in converting/displaying as jpeg
    # Store training images as np array (N, H, W) with range [0,255], distances and angles as (N,)
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


def partial_dataset(dataset, index_range):
    start = index_range[0]
    finish = index_range[1]
    sub_dataset = {}
    sub_dataset['index'] = np.array(range(start, finish))
    for key in ['image_upper_bounds', 'image_lower_bounds', 'images']:
        sub_dataset[key] = dataset[key][start:finish,:,:]
    for key in ['distance_upper_bounds', 'distance_lower_bounds', 'angle_lower_bounds', 'angle_upper_bounds', 'distances', 'angles']:
        sub_dataset[key] = dataset[key][start:finish]

    return sub_dataset

def generate_partial_dataset(shape, distance_range, angle_range, distance_grid_num, angle_grid_num, index_range, thread_num, save_dir, noise_mode, noise_scale):
    # distance_range and angle_range are list, [low, high]
    # Get distance center and deltas, uniform on inverse scale
    d_inv_start = 1.0 / distance_range[1]
    d_inv_end = 1.0 / distance_range[0]
    d_inv_step = (d_inv_end - d_inv_start) / distance_grid_num
    d_inv_bounds = np.arange(d_inv_start, d_inv_end + 1e-5, d_inv_step)
    d_bounds = np.flip(1.0/d_inv_bounds)
    assert len(d_bounds) == distance_grid_num + 1, "distance bounds count incorrect"

    d_centers = [(d_bounds[i] + d_bounds[i+1])/2 for i in range(distance_grid_num)]
    d_deltas = [(d_bounds[i+1] - d_bounds[i])/2 for i in range(distance_grid_num)]

    angle_grid_size = (angle_range[1] - angle_range[0])/angle_grid_num
    angle_delta = angle_grid_size / 2
    image_lower_bounds = []
    image_upper_bounds = []
    distance_lower_bounds = []
    distance_upper_bounds = []
    angle_lower_bounds = []
    angle_upper_bounds = []
    start = index_range[0]
    finish = index_range[1]
    sub_dataset = {}
    sub_dataset['index'] = np.array(range(start, finish))
    sensor = get_sensor(shape, noise_mode, noise_scale)
    for index in tqdm(range(start, finish)):
        i = index // distance_grid_num
        j = index % distance_grid_num
        distance = d_centers[j]
        distance_delta = d_deltas[j]
        angle = angle_range[0] + i * angle_grid_size + angle_delta
        lower_bound_matrix, upper_bound_matrix = sensor.take_measurement_with_range(distance, angle, distance_delta, angle_delta)
        image_lower_bounds.append(np.expand_dims(lower_bound_matrix, axis=0))
        image_upper_bounds.append(np.expand_dims(upper_bound_matrix, axis=0))
        distance_lower_bounds.append(distance - distance_delta)
        distance_upper_bounds.append(distance + distance_delta)
        angle_lower_bounds.append(angle - angle_delta)
        angle_upper_bounds.append(angle + angle_delta)

    sub_dataset['image_lower_bounds'] = np.concatenate(image_lower_bounds, axis=0) # (N, H, W)
    sub_dataset['image_upper_bounds'] = np.concatenate(image_upper_bounds, axis=0) # (N, H, W)
    sub_dataset['distance_lower_bounds'] = np.array(distance_lower_bounds) # (N,)
    sub_dataset['distance_upper_bounds'] = np.array(distance_upper_bounds) # (N,)
    sub_dataset['angle_lower_bounds'] = np.array(angle_lower_bounds) # (N,)
    sub_dataset['angle_upper_bounds'] = np.array(angle_upper_bounds) # (N,)
    sub_dataset['distance_grid_num'] = distance_grid_num
    sub_dataset['angle_grid_num'] = angle_grid_num

    sio.savemat(os.path.join(save_dir, '{}_thread_{}.mat'.format(shape.value, thread_num)), sub_dataset)

    return True


def gen_data_for_verify_parallel(distance_min, distance_max, angle_rng, grid_size, num_threads, noise_mode, noise_scale):
    distance_range = [distance_min, distance_max]
    angle_range = [-angle_rng, angle_rng]
    # Distance use finer grain
    distance_grid_size = grid_size/2
    angle_grid_size = grid_size
    distance_grid_num = int((distance_max - distance_min)/distance_grid_size)
    angle_grid_num = int(2*angle_rng/angle_grid_size)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)

    save_dir = os.path.join(data_dir, 'lidar_distance_min_{}_max_{}_angle_{}_grid_{}_thread_{}'.format(distance_min, distance_max, angle_rng, grid_size, num_threads)+noise_mode+'{}_small'.format(noise_scale))
    if not os.path.exists(save_dir):
        print("Creating {}".format(save_dir))
        os.makedirs(save_dir)

    if os.path.isfile(os.path.join(save_dir, 'info.mat')):
        print('Dataset for verify is already generated, skip the generation')
    else:
        # Store the grid num and range information
        info = {}
        info['distance_grid_num'] = distance_grid_num
        info['angle_grid_num'] = angle_grid_num
        info['distance_min'] = distance_min
        info['distance_max'] = distance_max
        info['angle_range'] = angle_rng
        sio.savemat(os.path.join(save_dir, 'info.mat'), info)

        # Split the dataset to separate files
        N = distance_grid_num * angle_grid_num
        num_threads_per_class = num_threads//3
        n_per_core = N//num_threads_per_class +1
        range_list = [(i*n_per_core, min((i+1)*n_per_core, N)) for i in range(num_threads_per_class)]
        pool = mp.Pool(processes=num_threads)
        results = {}
        for shape in Shape:
            results[shape] = [pool.apply_async(generate_partial_dataset, args=(shape, distance_range, angle_range, distance_grid_num, angle_grid_num, index_range, thread_num, save_dir, noise_mode, noise_scale)) for (thread_num, index_range) in enumerate(range_list)]

        for shape in Shape:
            for i in range(num_threads_per_class):
                retval = results[shape][i].get()


def main():
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--mode', type=str, help='Mode of generation, currently support train')
    parser.add_argument('--distance_max_train', type=int, help='Max distance')
    parser.add_argument('--distance_min_train', type=int, help='Min distance')
    parser.add_argument('--angle_range_train', type=int, help='Range for angle')
    parser.add_argument('--distance_max_valid', type=int, help='Max distance')
    parser.add_argument('--distance_min_valid', type=int, help='Min distance')
    parser.add_argument('--angle_range_valid', type=int, help='Range for angle')
    parser.add_argument('--target_dir_name', type=str, help='Directory name to save the generated data')
    # For estimate mode
    parser.add_argument('--grid_size', type=float, default=0.1, help='Grid size for calculating error')
    # For train mode
    parser.add_argument('--noise', type=str, default='none', help='Noise mode, can be none/uniform/gaussian')
    parser.add_argument('--noise_scale', type=float, default=0.05, help='Scale of noise, for uniform it is the max, for gaussian it is one sigma')
    args = parser.parse_args()

    if args.mode == 'estimate':
        #gen_data_for_estimate(args.distance_range, args.angle_range, args.grid_size, args.target_dir_name, args.noise, args.noise_scale)
        pass
    elif args.mode == 'train':
        gen_train_valid_data(args.distance_min_train, args.distance_max_train, args.angle_range_train, args.distance_min_valid, args.distance_max_valid, args.angle_range_valid, args.noise, args.noise_scale, args.target_dir_name)
    else:
        print("Generation mode not supported.")


    #sensor = get_sensor()
    #gen_example_picture(sensor)

    #gen_example_picture_with_range(sensor)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
