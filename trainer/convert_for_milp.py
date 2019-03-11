import torch
import torch.nn as nn

import argparse
import os

import scipy.io as sio
import numpy as np

from train import RESULTS_ROOT

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# cnn_small_torch corresponds to CNN_small in model.py
cnn_small_torch = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(6*6*32, 100),
        nn.ReLU(),
        nn.Linear(100, 2)
    )

def main():
    parser = argparse.ArgumentParser(description='convert .pth checkpoint file from pytorch training for MIPVerify.')
    parser.add_argument('name', help='name of experiment to convert the model parameters')
    args = parser.parse_args()
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
    parameters_torch["logits/weight"] = np.transpose(cnn_small_torch[7].weight.data.numpy())
    parameters_torch["logits/bias"] = cnn_small_torch[7].bias.data.numpy()
    converted_file_path = os.path.join(exp_dir, 'converted.mat')
    sio.savemat(converted_file_path, parameters_torch)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
