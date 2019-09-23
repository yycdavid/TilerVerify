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
import matplotlib.pyplot as plt

def show_adv_inputs_images(adv_input, label, target_label, angle_max, angle_min, distance_max, distance_min, logit_diff, save_dir):
    plt.figure()
    plt.imshow(np.transpose(np.squeeze(adv_input)), cmap='rainbow_r')
    cb = plt.colorbar(extend='max')
    plt.axis('off')
    save_path = os.path.join(save_dir, 'label_{}_to_{}_angle_{:.3f}_{:.3f}_dist_{:.3f}_{:.3f}_diff_{:.1f}.jpg'.format(label, target_label, angle_min, angle_max, distance_min, distance_max, logit_diff))
    plt.savefig(save_path, bbox_inches='tight')



def main():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    exp_dir = os.path.join(data_dir, 'lidar_distance_min_30_max_60_angle_45_grid_0.5_thread_21gaussian0.001_small')
    adv_dir = os.path.join(exp_dir, 'adv_inputs')

    # Read adversarial inputs
    verify_file_path = os.path.join(adv_dir, 'adv_inputs.mat')
    with h5py.File(verify_file_path, 'r') as f:
        verify_result = {}
        for k, v in f.items():
            verify_result[k] = np.array(v)
        perturbedInputs = [f[obj_ref].value for obj_ref in verify_result['perturbedInput']]
        verify_result['perturbedInput'] = perturbedInputs

    # Plot adversarial inputs
    for i in range(len(verify_result['label'])):
        show_adv_inputs_images(verify_result['perturbedInput'][i], verify_result['label'][i], verify_result['targetLabel'][i], verify_result['angleMax'][i], verify_result['angleMin'][i], verify_result['distanceMax'][i], verify_result['distanceMin'][i], verify_result['logitDiff'][i], adv_dir)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
