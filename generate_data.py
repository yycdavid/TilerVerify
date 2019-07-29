import os
import image_generator
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
import multiprocessing as mp

# 1 Unit = 5 centimeters
scene_params = {
'line_width': 4.0,
'road_width': 50.0, # per lane
'shade_width': 1.0,
}

camera_params = {
'height': 20.0,
'focal_length': 1.0,
'pixel_num': 32,
'pixel_size': 0.16,
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

def generate_dataset_for_error_est_parallel(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side):
    # offset_range and angle_range are list, [low, high]
    # returned dataset has order: innermost is within a grid, then varying angle, then varying offset
    num_threads = 30
    N = offset_grid_num * angle_grid_num * num_points_per_side * num_points_per_side
    n_per_core = N//num_threads +1
    range_list = [(i*n_per_core, min((i+1)*n_per_core, N)) for i in range(num_threads)]
    pool = mp.Pool(processes=num_threads)
    results = [pool.apply_async(generate_partial_est_dataset, args=(offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side, index_range)) for index_range in range_list]
    sub_datasets = []
    for i in range(num_threads):
        sub_datasets.append(results[i].get())
    for i in range(1, num_threads):
        assert sub_datasets[i]['index'][0] > sub_datasets[i-1]['index'][0], "Result is out of order"

    dataset = {}
    dataset['images'] = np.concatenate([sub_datasets[i]['images'] for i in range(num_threads)], axis=0)
    dataset['offsets'] = np.concatenate([sub_datasets[i]['offsets'] for i in range(num_threads)])
    dataset['angles'] = np.concatenate([sub_datasets[i]['angles'] for i in range(num_threads)])
    dataset['points_per_grid'] = num_points_per_side * num_points_per_side
    dataset['offset_grid_num'] = offset_grid_num
    dataset['angle_grid_num'] = angle_grid_num
    return dataset

def generate_partial_est_dataset(offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side, index_range):
    # offset_range and angle_range are list, [low, high]
    offset_grid_size = (offset_range[1] - offset_range[0])/offset_grid_num
    angle_grid_size = (angle_range[1] - angle_range[0])/angle_grid_num
    offset_point_space = offset_grid_size / num_points_per_side
    angle_point_space = angle_grid_size / num_points_per_side
    points_per_grid = num_points_per_side * num_points_per_side
    images = []
    offsets = []
    angles = []
    start = index_range[0]
    finish = index_range[1]
    sub_dataset = {}
    sub_dataset['index'] = np.array(range(start, finish))
    viewer = get_viewer()
    for index in tqdm(range(start, finish)):
        index_ij = index // points_per_grid
        index_pq = index % points_per_grid
        p = index_pq // num_points_per_side
        q = index_pq % num_points_per_side
        i = index_ij // angle_grid_num
        j = index_ij % angle_grid_num
        offset = offset_range[0] + i * offset_grid_size + p * offset_point_space + offset_point_space/2
        angle = angle_range[0] + j * angle_grid_size + q * angle_point_space + angle_point_space/2
        images.append(np.expand_dims(viewer.take_picture(offset, angle), axis=0))
        offsets.append(offset)
        angles.append(angle)

    sub_dataset['images'] = np.concatenate(images, axis=0) # (N, H, W)
    sub_dataset['offsets'] = np.array(offsets) # (N,)
    sub_dataset['angles'] = np.array(angles) # (N,)

    return sub_dataset


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

def gen_train_valid_data(offset_range, angle_range, noise_mode, noise_scale, target_dir_name):
    viewer = get_viewer(noise_mode, noise_scale)
    # Generate training and validation dataset
    offset_range = [-offset_range, offset_range]
    angle_range = [-angle_range, angle_range]
    training_size = 130000
    validation_size = 1000
    training_set = generate_dataset(viewer, training_size, offset_range, angle_range)
    validation_set = generate_dataset(viewer, validation_size, offset_range, angle_range)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', target_dir_name)
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)
    sio.savemat(os.path.join(data_dir, 'train_bigger_130000.mat'), training_set)
    sio.savemat(os.path.join(data_dir, 'valid_bigger_1000.mat'), validation_set)

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
    offset = 10.0
    angle = 15.0
    image_taken = viewer.take_picture(offset, angle)

    img = Image.fromarray(image_taken)
    img = img.convert("L")
    img.save('32_offset_{}_angle_{}.pdf'.format(offset, angle))

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


def get_viewer(noise_mode='none', noise_scale=0.0):
    scene = image_generator.Scene(scene_params)
    viewer = image_generator.Viewer(camera_params, scene, noise_mode, noise_scale)
    return viewer

def partial_dataset(dataset, index_range):
    start = index_range[0]
    finish = index_range[1]
    sub_dataset = {}
    sub_dataset['index'] = np.array(range(start, finish))
    for key in ['image_upper_bounds', 'image_lower_bounds', 'images']:
        sub_dataset[key] = dataset[key][start:finish,:,:]
    for key in ['offset_upper_bounds', 'offset_lower_bounds', 'angle_lower_bounds', 'angle_upper_bounds', 'offsets', 'angles']:
        sub_dataset[key] = dataset[key][start:finish]

    return sub_dataset

def generate_partial_dataset(offset_range, angle_range, offset_grid_num, angle_grid_num, index_range, thread_num, save_dir):
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
    start = index_range[0]
    finish = index_range[1]
    sub_dataset = {}
    sub_dataset['index'] = np.array(range(start, finish))
    viewer = get_viewer()
    for index in tqdm(range(start, finish)):
        i = index // angle_grid_num
        j = index % angle_grid_num
        offset = offset_range[0] + i * offset_grid_size + offset_delta
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

    sub_dataset['image_lower_bounds'] = np.concatenate(image_lower_bounds, axis=0) # (N, H, W)
    sub_dataset['image_upper_bounds'] = np.concatenate(image_upper_bounds, axis=0) # (N, H, W)
    sub_dataset['offset_lower_bounds'] = np.array(offset_lower_bounds) # (N,)
    sub_dataset['offset_upper_bounds'] = np.array(offset_upper_bounds) # (N,)
    sub_dataset['angle_lower_bounds'] = np.array(angle_lower_bounds) # (N,)
    sub_dataset['angle_upper_bounds'] = np.array(angle_upper_bounds) # (N,)
    sub_dataset['images'] = np.concatenate(images, axis=0) # (N, H, W)
    sub_dataset['offsets'] = np.array(offsets) # (N,)
    sub_dataset['angles'] = np.array(angles) # (N,)
    sub_dataset['offset_grid_num'] = offset_grid_num
    sub_dataset['angle_grid_num'] = angle_grid_num

    sio.savemat(os.path.join(save_dir, 'thread_{}.mat'.format(thread_num)), sub_dataset)

    return True

'''
def gen_data_for_verify_parallel(offset_rng, angle_rng, grid_size, num_threads):
    viewer = get_viewer()
    offset_range = [-offset_rng, offset_rng]
    angle_range = [-angle_rng, angle_rng]
    offset_grid_num = int(2*offset_rng/grid_size)
    angle_grid_num = int(2*angle_rng/grid_size)
    dataset = generate_dataset_for_verify(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)

    save_dir = os.path.join(data_dir, 'verify_offset_{}_angle_{}_grid_{}_thread_{}'.format(offset_rng, angle_rng, grid_size, num_threads))
    if not os.path.exists(save_dir):
        print("Creating {}".format(save_dir))
        os.makedirs(save_dir)

    if os.path.isfile(os.path.join(save_dir, 'info.mat')):
        print('Dataset for verify is already generated, skip the generation')
    else:
        # Split the dataset to separate files
        N = dataset['offsets'].shape[0]
        n_per_core = N//num_threads +1
        range_list = [(i*n_per_core, min((i+1)*n_per_core, N)) for i in range(num_threads)]
        for i in range(num_threads):
            sub_dataset = partial_dataset(dataset, range_list[i])
            sio.savemat(os.path.join(save_dir, 'thread_{}.mat'.format(i)), sub_dataset)
        # Store the grid num and range information
        info = {}
        info['offset_grid_num'] = dataset['offset_grid_num']
        info['angle_grid_num'] = dataset['angle_grid_num']
        info['offset_range'] = offset_rng
        info['angle_range'] = angle_rng
        sio.savemat(os.path.join(save_dir, 'info.mat'), info)
'''

def gen_data_for_verify_parallel(offset_rng, angle_rng, grid_size, num_threads):
    offset_range = [-offset_rng, offset_rng]
    angle_range = [-angle_rng, angle_rng]
    offset_grid_num = int(2*offset_rng/grid_size)
    angle_grid_num = int(2*angle_rng/grid_size)

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)

    save_dir = os.path.join(data_dir, 'verify_offset_{}_angle_{}_grid_{}_thread_{}'.format(offset_rng, angle_rng, grid_size, num_threads))
    if not os.path.exists(save_dir):
        print("Creating {}".format(save_dir))
        os.makedirs(save_dir)

    if os.path.isfile(os.path.join(save_dir, 'info.mat')):
        print('Dataset for verify is already generated, skip the generation')
    else:
        # Store the grid num and range information
        info = {}
        info['offset_grid_num'] = offset_grid_num
        info['angle_grid_num'] = angle_grid_num
        info['offset_range'] = offset_rng
        info['angle_range'] = angle_rng
        sio.savemat(os.path.join(save_dir, 'info.mat'), info)

        # Split the dataset to separate files
        N = offset_grid_num * angle_grid_num
        n_per_core = N//num_threads +1
        range_list = [(i*n_per_core, min((i+1)*n_per_core, N)) for i in range(num_threads)]
        pool = mp.Pool(processes=num_threads)
        results = [pool.apply_async(generate_partial_dataset, args=(offset_range, angle_range, offset_grid_num, angle_grid_num, index_range, thread_num, save_dir)) for (thread_num, index_range) in enumerate(range_list)]
        for i in range(num_threads):
            retval = results[i].get()

def gen_data_for_estimate(offset_rng, angle_rng, grid_size, target_dir_name):
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print("Creating {}".format(data_dir))
        os.makedirs(data_dir)

    save_dir = os.path.join(data_dir, target_dir_name)
    if not os.path.exists(save_dir):
        print("Creating {}".format(save_dir))
        os.makedirs(save_dir)

    # Generate a test set for error estimation
    offset_range = [-offset_rng, offset_rng]
    angle_range = [-angle_rng, angle_rng]
    offset_grid_num = int(2*offset_rng/grid_size)
    angle_grid_num = int(2*angle_rng/grid_size)
    num_points_per_side = math.ceil(grid_size/0.05)
    viewer = get_viewer()
    dataset = generate_dataset_for_error_est_parallel(viewer, offset_range, angle_range, offset_grid_num, angle_grid_num, num_points_per_side)

    with open(os.path.join(save_dir, 'error_estimate_data.pickle'), 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)




def main():
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--mode', type=str, help='Mode of generation, currently support estimate/train')
    parser.add_argument('--offset_range', type=int, help='Range for offset')
    parser.add_argument('--angle_range', type=int, help='Range for angle')
    parser.add_argument('--target_dir_name', type=str, help='Directory name to save the generated data')
    # For estimate mode
    parser.add_argument('--grid_size', type=float, default=0.1, help='Grid size for calculating error')
    # For train mode
    parser.add_argument('--noise', type=str, default='none', help='Noise mode, can be none/uniform/gaussian')
    parser.add_argument('--noise_scale', type=float, default=0.05, help='Scale of noise, for uniform it is the max, for gaussian it is one sigma')
    args = parser.parse_args()

    if args.mode == 'estimate':
        gen_data_for_estimate(args.offset_range, args.angle_range, args.grid_size, args.target_dir_name)
    elif args.mode == 'train':
        gen_train_valid_data(args.offset_range, args.angle_range, args.noise, args.noise_scale, args.target_dir_name)
    else:
        print("Generation mode not supported.")


    #viewer = get_viewer()
    #gen_example_picture(viewer)

    #gen_example_picture_with_range(viewer)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
