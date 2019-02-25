# TilerVerify
Highest level system:
- Scene generator (Python)
    - Core function: given a parameter pair (x, phi), generate a picture, store (pixel matrix, range for each pixel, ground truth tuple, range for ground truth tupel); or just store (pixel matrix, ground truth tuple), for training models
    - Structue: a class represent the world, a class represent the viewer (camera)
        To get an image, the main program tells the viewer (x,phi), and their range, the viewer computes for each pixel the ray intercept, and the shade covered in the range, use it to query the world to get intensity value and range

- Model training (Pytorch)
    - Core function: train a model with the synthetic data, potentially need adversarial training. In addition, output the trained model parameters to a format usable by MILP system
    -

- MILP evaluation (Julia)
    - Core function: Take in a trained model, a set of test points with range, produce a max error for each test point

