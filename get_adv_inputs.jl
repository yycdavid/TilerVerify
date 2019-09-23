include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

# Argument: 1. exp_name 2. data_name 3. thread_number
exp_name = "lidar_small_gaussian_0.001"
data_name = "lidar_distance_min_30_max_60_angle_45_grid_0.5_thread_21gaussian0.001_small"

# Get info from data_name

nnparams = get_custom_network_params("CNN_lidar", exp_name)

MIPVerify.setloglevel!("info")

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

#test_dataset = read_lidar_test_dataset("data/lidar_train_gaussian_0.001/valid.pickle")
#println("Fraction correct of validation set is $(frac_correct(nnparams, test_dataset, 2000))")

test_dataset_with_range = read_lidar_dataset_target("data/$(data_name)/adv_inputs/target_boxes.mat")


MIPVerify.find_adversarial_example_lidar(
    nnparams,
    test_dataset_with_range,
    GurobiSolver(Gurobi.Env(), TimeLimit=10),
    save_path = joinpath("data", data_name, "adv_inputs"),
    pp = MIPVerify.CustomPerturbationFamily(),
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=mip,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=10),
)
