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
    offset = 50.0
    angle = 15.0
    image_taken = viewer.take_picture(offset, angle)
    img = Image.fromarray(image_taken)
    img = img.convert("L")
    img.save('test.jpg')



if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.set_trace()
