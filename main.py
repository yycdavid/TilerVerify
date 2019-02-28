import os
import image_generator
from PIL import Image
import numpy as np

# Unit centimeters
scene_params = {
'line_width': 30.0,
'road_width': 300.0, # per lane
'shade_width': 5.0,
}

camera_params = {
'height': 150.0,
'focal_length': 5.0,
'pixel_num': 256,
'pixel_size': 0.1,
}

def main():
    scene = image_generator.Scene(scene_params)
    viewer = image_generator.Viewer(camera_params, scene)
    offset = 30.0
    angle = 0.0
    #image_taken = viewer.take_picture(offset, angle)

    delta_x = 2.0
    delta_phi = 1.0
    image_matrix, lower_bound_matrix, upper_bound_matrix = viewer.take_picture_with_range(offset, angle, delta_x, delta_phi)
    # Don't need the following to generate data, just visualization. There is potentially a alias in converting/displaying as jpeg
    img = Image.fromarray(image_matrix)
    img = img.convert("L")
    img.save('test.jpg')
    img_low = Image.fromarray(lower_bound_matrix)
    img_low = img_low.convert("L")
    img_low.save('test_low.jpg')
    img_upper = Image.fromarray(upper_bound_matrix)
    img_upper = img_upper.convert("L")
    img_upper.save('test_upper.jpg')


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.set_trace()
