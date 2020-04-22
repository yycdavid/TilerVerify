import os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
import time
import torch
import torch.nn as nn
from collections import OrderedDict

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not base_dir in sys.path:
    sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'trainer'))
from generate_data import get_viewer, get_new_viewer
import utils

from trainer.dataset import RoadSceneDataset

outputManager = 0

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
        outputManager.say("Loading bounding boxes...")
        with open(file_path, 'rb') as f:
            bounding_boxes = pickle.load(f)
        with open(os.path.join(data_dir, 'ground_truth_bounds.mat'), 'rb') as f:
            ground_truth_bounds = pickle.load(f)
    else:
        outputManager.say("Creating bounding boxes...")
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


def get_input_image(mode):
    if mode == 'in':
        viewer = get_viewer()
    elif mode == 'noise':
        viewer = get_viewer(noise_mode='uniform', noise_scale=0.1)
    else:
        viewer = get_new_viewer()
    offset = np.random.uniform(low=-40.0, high=40.0)
    angle = np.random.uniform(low=-60.0, high=60.0)
    #outputManager.say("Example image offset: {}, angle: {}".format(offset, angle))
    example_image = viewer.take_picture(offset, angle)
    return example_image, offset, angle


class InputDetector(object):
    """docstring for InputDetector."""

    def __init__(self, lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds, offset_grid_size, angle_grid_size, offset_error_bound, angle_error_bound):
        super(InputDetector, self).__init__()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.offset_lower_bounds = offset_lower_bounds
        self.angle_lower_bounds = angle_lower_bounds
        self.offset_grid_size = offset_grid_size
        self.angle_grid_size = angle_grid_size
        self.offset_error_bound = offset_error_bound
        self.angle_error_bound = angle_error_bound

    def detect_input(self, input_image):
        return np.any(np.logical_and(np.all(self.lower_bounds <= input_image, axis=1), np.all(input_image <= self.upper_bounds, axis=1)))

    def detect_input_with_prediction(self, input_image, offset_pred, angle_pred):
        indices = (self.offset_lower_bounds >= offset_pred - self.offset_error_bound - self.offset_grid_size) & \
            (self.offset_lower_bounds <= offset_pred + self.offset_error_bound) & \
            (self.angle_lower_bounds >= angle_pred - self.angle_error_bound - self.angle_grid_size) & \
            (self.angle_lower_bounds <= angle_pred + self.angle_error_bound)

        return np.any(np.logical_and(np.all(self.lower_bounds[indices] <= input_image, axis=1), np.all(input_image <= self.upper_bounds[indices], axis=1)))


def prepare_test_dataset(num_images, mode='in'):
    # Get an input image
    outputManager.say('Get example images...')
    example_images = []
    offsets = []
    angles = []
    for i in range(num_images):
        example_image, offset, angle = get_input_image(mode)
        example_images.append(np.expand_dims(example_image, axis=0))
        offsets.append(offset)
        angles.append(angle)

    dataset = {}
    dataset['images'] = np.concatenate(example_images, axis=0) # (N, H, W)
    dataset['offsets'] = np.array(offsets) # (N,)
    dataset['angles'] = np.array(angles) # (N,)
    test_dataset = RoadSceneDataset(dataset['images'], np.squeeze(dataset['offsets']), np.squeeze(dataset['angles']))

    example_images = [np.reshape(example_image, (1, -1)) for example_image in example_images]
    return test_dataset, example_images


def load_trained_nn():
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    exp_dir = os.path.join(base_dir, 'trained_models', 'big_130000')
    model_file_path = os.path.join(exp_dir, 'best_model.pt')
    assert os.path.isfile(model_file_path), "The experiment required has not been run or does not have the model trained."

    # cnn_small_torch corresponds to CNN_small in model.py
    cnn_small_torch = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 16, 4, stride=2, padding=1)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(16, 32, 4, stride=2, padding=1)),
        ('relu2', nn.ReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear(8*8*32, 100)),
        ('relu3', nn.ReLU()),
        ('fc2', nn.Linear(100, 2))
    ]))
    cnn_small_torch.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    return cnn_small_torch


def get_predictions(model, test_loader, device):
    model.eval()
    offset_preds = []
    angle_preds = []
    with torch.no_grad():
        for images, offsets, angles in tqdm(test_loader):
            images = images.to(device)
            output = model(images).numpy()
            offset_preds.append(output[:,0])
            angle_preds.append(output[:,1])
    offset_preds = np.concatenate(offset_preds)
    angle_preds = np.concatenate(angle_preds)
    return offset_preds, angle_preds


def get_example_images():
    viewer = get_viewer()
    offset = -10.0
    angle = 10.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('in1.pdf')

    offset = 10.0
    angle = 0.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('in2.pdf')

    viewer = get_viewer(noise_mode='uniform', noise_scale=0.2)
    offset = -20.0
    angle = 10.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('noise1.pdf')

    offset = 0.0
    angle = -10.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('noise2.pdf')

    viewer = get_new_viewer()
    offset = 0.0
    angle = 0.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('new1.pdf')

    offset = 20.0
    angle = -20.0
    example_image = viewer.take_picture(offset, angle)
    img = Image.fromarray(example_image)
    img = img.convert("L")
    img.save('new2.pdf')


def main():
    parser = argparse.ArgumentParser(description='Detecting whether input is legal or not')
    parser.add_argument('--file_name', type=str, help='File path to bounding boxes')
    args = parser.parse_args()

    result_dir = os.path.dirname(os.path.realpath(__file__))
    global outputManager
    outputManager = utils.OutputManager(result_dir)
    # Load bounding boxes
    lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds = get_bounding_boxes(args.file_name, 'naive')

    offset_grid_size = angle_grid_size = 0.2
    offset_error_bound = 11.058546699918972
    angle_error_bound = 5.269179741339784

    input_detector = InputDetector(lower_bounds, upper_bounds, offset_lower_bounds, angle_lower_bounds, offset_grid_size, angle_grid_size, offset_error_bound, angle_error_bound)


    # Prepare dataset for test
    for mode in ['in', 'noise', 'out']:
        outputManager.say("Testing mode " + mode + " start:")
        num_images = 500
        test_dataset, example_images = prepare_test_dataset(num_images, mode)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1, shuffle=False)

        # Get network predictions
        device = torch.device("cpu")
        torch.set_num_threads(1)
        nn_model = load_trained_nn()
        outputManager.say("Start predicting...")
        true_count = 0
        start_t = time.time()
        offset_preds, angle_preds = get_predictions(nn_model, test_loader, device)
        # To make running time comparison fair
        for i in range(num_images):
            true_count += 1
        end_t = time.time()
        outputManager.say('Time spent per input in inference: {}'.format((end_t - start_t)/num_images))

        # Decide if it's in any of the bounding box, naive method
        outputManager.say("Start detecting (naive)...")
        true_count = 0
        start_t = time.time()
        for i in range(num_images):
            is_legal = input_detector.detect_input(example_images[i])
            true_count += is_legal
        end_t = time.time()
        outputManager.say('Time spent per input (naive): {}'.format((end_t - start_t)/num_images))
        outputManager.say('{} out of {} detected as legal'.format(true_count, num_images))

        # Decide if it's in any of the bounding box, guided search
        outputManager.say("Start detecting (guided)...")
        true_count = 0
        start_t = time.time()
        for i in range(num_images):
            is_legal = input_detector.detect_input_with_prediction(example_images[i], offset_preds[i], angle_preds[i])
            true_count += is_legal
        end_t = time.time()
        outputManager.say('Time spent per input (guided): {}'.format((end_t - start_t)/num_images))
        outputManager.say('{} out of {} detected as legal'.format(true_count, num_images))


if __name__ == '__main__':
    try:
        get_example_images()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
                                                                                                                                                                                                                                                                                                                                      
