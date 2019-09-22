import torch
import torch.nn as nn
from collections import OrderedDict
import argparse
import os

import scipy.io as sio
import numpy as np

from train import RESULTS_ROOT

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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


cnn_lidar = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 16, 4, stride=2, padding=1)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(16, 16, 4, stride=2, padding=1)),
        ('relu2', nn.ReLU()),
        #('conv3', nn.Conv2d(16, 32, 4, stride=2, padding=1)),
        #('relu3', nn.ReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear(8*8*16, 100)),
        ('relu4', nn.ReLU()),
        ('fc2', nn.Linear(100, 3))
    ]))


def main_for_road(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, args.name)
    model_file_path = os.path.join(exp_dir, 'best_model.pt')
    assert os.path.isfile(model_file_path), "The experiment required has not been run or does not have the model trained."

    cnn_small_torch.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    parameters_torch=dict()
    # transposing the tensor is necessary because pytorch and Julia have different conventions.
    parameters_torch["conv1/weight"] = np.transpose(cnn_small_torch[0].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv1/bias"] = cnn_small_torch[0].bias.data.numpy()
    parameters_torch["conv2/weight"] = np.transpose(cnn_small_torch[2].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv2/bias"] = cnn_small_torch[2].bias.data.numpy()
    parameters_torch["fc1/weight"] = np.transpose(cnn_small_torch[5].weight.data.numpy())
    parameters_torch["fc1/bias"] = cnn_small_torch[5].bias.data.numpy()
    parameters_torch["fc2/weight"] = np.transpose(cnn_small_torch[7].weight.data.numpy())
    parameters_torch["fc2/bias"] = cnn_small_torch[7].bias.data.numpy()
    converted_file_path = os.path.join(exp_dir, 'converted.mat')
    sio.savemat(converted_file_path, parameters_torch)


def main_for_lidar(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, args.name)
    model_file_path = os.path.join(exp_dir, 'best_model.pt')
    assert os.path.isfile(model_file_path), "The experiment required has not been run or does not have the model trained."

    cnn_lidar.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    parameters_torch=dict()
    # transposing the tensor is necessary because pytorch and Julia have different conventions.
    parameters_torch["conv1/weight"] = np.transpose(cnn_lidar[0].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv1/bias"] = cnn_lidar[0].bias.data.numpy()
    parameters_torch["conv2/weight"] = np.transpose(cnn_lidar[2].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv2/bias"] = cnn_lidar[2].bias.data.numpy()
    #parameters_torch["conv3/weight"] = np.transpose(cnn_lidar[4].weight.data.numpy(), [2, 3, 1, 0])
    #parameters_torch["conv3/bias"] = cnn_lidar[4].bias.data.numpy()
    parameters_torch["fc1/weight"] = np.transpose(cnn_lidar[5].weight.data.numpy())
    parameters_torch["fc1/bias"] = cnn_lidar[5].bias.data.numpy()
    parameters_torch["fc2/weight"] = np.transpose(cnn_lidar[7].weight.data.numpy())
    parameters_torch["fc2/bias"] = cnn_lidar[7].bias.data.numpy()
    converted_file_path = os.path.join(exp_dir, 'converted.mat')
    sio.savemat(converted_file_path, parameters_torch)


def get_args():
    parser = argparse.ArgumentParser(description='convert .pth checkpoint file from pytorch training for MIPVerify.')
    parser.add_argument('--name', help='name of experiment to convert the model parameters')
    parser.add_argument('--case', type=str, help='Case study to convert. Can be: road/lidar')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.case == 'road':
        main_for_road(args)
    elif args.case == 'lidar':
        main_for_lidar(args)
    else:
        raise ValueError("Only support case study road or lidar")


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
