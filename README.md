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
- What we want to show from this experiment:
How good the error bound is: 1) how close it is to the true error 2) is it in phase with true error landscape
Can you make the error bound tighter in this framework: grid size against error

In a reasonable setting and a reasonable input space, our method can give a bound of 5 percent of the input
range.  

(3.25-3.30)
- Setup the experiment on clusters, use parallelism (3.25,26)
- Run experiments for a larger range (potentially a different scene and camera settings) (3.26)
- Measuring bound closeness by taking difference between bound and actual and divide by range for measurement. (3.27)
- Progressive tiling/adaptive grid size selection (3.27-30)

(4.1-7)
- Run experiments

(4.7-21)
- Write up
