# TilerVerify
Highest level system:
- Scene generator (Python)
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


The current settings runs about 9 seconds per test point; error is roughly +/- 10 points, for offset~30 and angle~5-10 with range of 1. Perturbed image looks natural.

TODO:
- Output the range dataset also with the center test point, in order for study purposes
- Write down our general framework, and think what we want to show from this experiment
- Write the batch processing, and result logging, before running any long-time experiments
