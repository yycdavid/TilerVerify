import os
import image_generator
import lidar_generator
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io as sio
import argparse
import math
import multiprocessing as mp
import matplotlib.pyplot as plt

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


# For Lidar, 1 Unit = 5 centimeters
scene_params_lidar = {
'stick_width': 1.0,
'stick_height': 40.0,
'shape': 'circle',
#'side_length': 10.0,
'radius': 5.0,
}

sensor_params_lidar = {
'height': 40.0,
'ray_num': 64,
'focal_length': 4.0,
'pixel_size': 0.1,
'max_distance': 300.0,
}

# measurement range: distance [20, 60], angle=[-45, 45]


def gen_example_picture_with_range(viewer):
    offset = 10
    angle = -10
    # Generate a picture with range
    delta_x = 0.1
    delta_phi = 0.1
    image_matrix, lower_bound_matrix, upper_bound_matrix = viewer.take_picture_with_range(offset, angle, delta_x, delta_phi)
    # Don't need the following to generate data, just visualization. There is potentially a alias in converting/displaying as jpeg
    # Store training images as np array (N, H, W) with range [0,255], offsets and angles as (N,)
    # Store pixel ranges as np array with range (0,1.0)
    img = Image.fromarray(image_matrix)
    img = img.convert("L")
    img.save('test.jpg')
    img_low = Image.fromarray(lower_bound_matrix)
    img_low = img_low.convert("L")
    img_low.save('test_lower.jpg')
    img_upper = Image.fromarray(upper_bound_matrix)
    img_upper = img_upper.convert("L")
    img_upper.save('test_upper.jpg')

# Get viewer in graphics
def get_viewer(noise_mode='none', noise_scale=0.0):
    scene = image_generator.Scene(scene_params)
    viewer = image_generator.Viewer(camera_params, scene, noise_mode, noise_scale)
    return viewer


def get_sensor(noise_mode='none', noise_scale=0.0):
    scene = lidar_generator.Scene(scene_params_lidar)
    sensor = lidar_generator.Sensor(sensor_params_lidar, scene, noise_mode, noise_scale)
    return sensor


def gen_example_lidar_measurement(sensor):
    angle = 0.0
    distance = 40.0
    pixel_matrix = sensor.take_measurement(angle, distance)

    plt.figure()

    plt.imshow(pixel_matrix, cmap='rainbow_r')
    cb = plt.colorbar(extend='max')
    plt.axis('off')
    #plt.clim(0, color_max);
    plt.savefig('test.jpg', bbox_inches='tight')

    '''img = Image.fromarray(pixel_matrix)
    img = img.convert("L")
    import pdb; pdb.set_trace()
    img.save('test.jpg')'''

def test_graphics():
    viewer = get_viewer('uniform', 0.01)
    gen_example_picture_with_range(viewer)


def test_lidar():
    sensor = get_sensor('gaussian', 0.005)
    gen_example_lidar_measurement(sensor)


def main():
    #test_graphics()
    test_lidar()

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
