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

- Generate data for estimating true error, estimate it (DONE)
- Plot error map as heat map (DONE)
- Implement the running and saving of verify, run it (DONE)
- Decide a range for input space to experiment on, then generate a new training set with more data. Train. Get estimated error, with more samples. Run MIP on this range. Compare error maps.
    - Range is [-10, 10] for both offset and angle. Train on [-20,20] for both, 10000 examples, train_20_10000.mat; validate on 500 examples, valid_20_500.mat; error estimate using grid size 1, test_error_est_10_1.mat; error bound using grid size 1, test_verify_10_1.mat 
- Implement statistics
- Generate a few datasets with different grid size, run, and measure the statistics for closeness

- Present: 1) Example images (also high resolution) 2) 2D heat maps 3) Statistics 4) Plot for different grid size

- Write the batch processing, and result logging, before running any long-time experiments
