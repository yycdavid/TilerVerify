# TilerVerify

This is the source code repository for NeurIPS 2019 submission XXX.

## Dependencies
The code is tested with Python 3.5.2 and Julia 0.6.4. Run dependency.jl to get required packages in Julia. Our code for solving neural network output range is based on [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl) (**Evaluating Robustness of Neural Networks with Mixed Integer Programming**
_Vincent Tjeng, Kai Xiao, Russ Tedrake_
https://arxiv.org/abs/1711.07356). Refer to this [documentation](https://vtjeng.github.io/MIPVerify.jl/latest) on setting up the dependencies for MIPVerify.jl (e.g. Gurobi optimizer).

## System Overview

- Image generator (Python)
    - Core function: given a parameter pair (x, phi), generate a picture, store (pixel matrix, range for each pixel, ground truth tuple, range for ground truth tupel); or just store (pixel matrix, ground truth tuple), for training models
    - Structue: a class represent the world, a class represent the viewer (camera)
        To get an image, the main program tells the viewer (x,phi), and their range, the viewer computes for each pixel the ray intercept, and the shade covered in the range, use it to query the world to get intensity value and range

- Model training (Pytorch)
    - Core function: train a model with the synthetic data, potentially need adversarial training. In addition, output the trained model parameters to a format usable by MILP system

- MILP evaluation (Julia)
    - Core function: Take in a trained model, a set of test points with range, produce a max error for each test point

To verify a new network:
1. Add the model in trainer/model.py, and train it
2. Add the model in trainer/convert_for_milp.py, and convert the trained model into correct format for MIPVerify
3. Add the model in MIPVerify/src/utils/import_example_nets.jl
