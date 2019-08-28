# TilerVerify

This is the source code repository for paper 'Correctness Verification of Neural Networks'. This is still updating with the research, so please contact Yichen Yang (yicheny@mit.edu) if you have issues using it.

## Dependencies
The code is tested with Python 3.5.2 and Julia 0.6.4. Run dependency.jl to get required packages in Julia. Our code for solving neural network output range is based on [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl) (**Evaluating Robustness of Neural Networks with Mixed Integer Programming**
_Vincent Tjeng, Kai Xiao, Russ Tedrake_
https://arxiv.org/abs/1711.07356). Refer to this [documentation](https://vtjeng.github.io/MIPVerify.jl/latest) on setting up the dependencies for MIPVerify.jl (e.g. Gurobi optimizer).

## Run experiment
Run run_exp.sh to reproduce the experiment. This script will:
- Train a CNN model (trainer/train.py)
- Convert the trained model to the format compatible with MIPVerify.jl (trainer/convert_for_milp.py)
- Compute bounding boxes for each tile and save them for solver (parallel_verify.py)
- Run solver from MIPVerify.jl to solve optimization problems (verify_thread.jl)
- Collect results and compute error upper bounds (thread_collect.jl)
- Compute error estimates (generate_data.py, trainer/error_estimate.py)
- Analyze results and produce plots (analysis/heatmap.py, analysis/statistics.py)

## System Overview

- Image generator (Python)
    - Core function: given a parameter pair (x, phi), generate a picture, store (pixel matrix, range for each pixel, ground truth tuple, range for ground truth tupel); or just store (pixel matrix, ground truth tuple), for training models
    - Structue: a class represent the world, a class represent the viewer (camera)
        To get an image, the main program tells the viewer (x,phi), and their range, the viewer computes for each pixel the ray intercept, and the region of sweep, use it to query the world to get intensity value and range

- Model training (Pytorch)
    - Core function: train a model with the synthetic data,. In addition, output the trained model parameters to a format usable by MILP system

- MILP solving (Julia)
    - Core function: Take in a trained model, a set of test points with range, produce a max error for each test point

To verify a new network:
1. Add the model in trainer/model.py, and train it
2. Add the model in trainer/convert_for_milp.py, and convert the trained model into correct format for MIPVerify
3. Add the model in MIPVerify/src/utils/import_example_nets.jl


Add functionality:
- Spec: Given a new image, decide whether it's within the bounding boxes or not.
- Test: 1) Legal images, 100% accept;
        2) Images with added elements (e.g. lines), able to reject
        3) Adversarially perturbed images, able to reject
- Engineering plan:
    Have a input_detector folder, deal with this
    Implement until a point when it can be run within a reasonable time

(100 samples, legal)
- NN inference: 0.0025s/input
- naive implementation: 1.138s/input
- guided: 0.069s/input

(20 samples, noisy)
- NN inference: 0.0048s/input
- naive: 0.979s/input
- guided: 0.090s/input


1000 samples
noisy: 0.0949s/input 1000 False
legal: 0.0431s/input 1000 True
different scene: 0.0528s/input, 1000 False
