import torch
import torch.nn as nn
import argparse
import os
import scipy.io as sio
from collections import OrderedDict
import numpy as np

from train import RESULTS_ROOT
from convert_for_milp import Flatten
from tqdm import tqdm
from dataset import RoadSceneDataset


def load_data(data_file):
    return sio.loadmat(data_file)

def compute_error(model, device, test_loader):
    model.eval()
    offset_errors = []
    angle_errors = []
    with torch.no_grad():
        for images, offsets, angles in tqdm(test_loader):
            images, offsets, angles = images.to(device), offsets.to(device).numpy(), angles.to(device).numpy()
            output = model(images).numpy()
            offset_errors.append(np.absolute(output[:,0] - offsets))
            angle_errors.append(np.absolute(output[:,1] - angles))
    offset_errors = np.concatenate(offset_errors)
    angle_errors = np.concatenate(angle_errors)
    return offset_errors, angle_errors

def main():
    parser = argparse.ArgumentParser(description='Compute estimated error on test set')
    parser.add_argument('--exp_name', help='name of experiment to compute error')
    parser.add_argument('--data', help='dataset file to compute error on')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    exp_dir = os.path.join(result_root, args.exp_name)
    model_file_path = os.path.join(exp_dir, 'best_model.pt')
    assert os.path.isfile(model_file_path), "The experiment required has not been run or does not have the model trained."

    device = torch.device("cpu")

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

    data_dir = os.path.join(base_dir, 'data', args.data)

    dataset = load_data(data_dir)

    test_dataset = RoadSceneDataset(dataset['images'], np.squeeze(dataset['offsets']), np.squeeze(dataset['angles']))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100, shuffle=False)

    offset_errors, angle_errors = compute_error(cnn_small_torch, device, test_loader)
    error_result = {}
    error_result['offset_errors'] = offset_errors
    error_result['angle_errors'] = angle_errors
    error_result['points_per_grid'] = dataset['points_per_grid']
    error_result['offset_grid_num'] = dataset['offset_grid_num']
    error_result['angle_grid_num'] = dataset['angle_grid_num']

    sio.savemat(os.path.join(exp_dir, 'error_est_result.mat'), error_result)




if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
