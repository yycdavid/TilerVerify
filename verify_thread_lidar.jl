include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

# Argument: 1. exp_name 2. data_name 3. thread_number
exp_name = ARGS[1]
data_name = ARGS[2]
shape = parse(Int64, ARGS[3])
thread_number = parse(Int64, ARGS[4])

# Get info from data_name
distance_min = parse(Int64, Base.split(data_name,"_")[4])
distance_max = parse(Int64, Base.split(data_name,"_")[6])
angle_range = parse(Int64, Base.split(data_name,"_")[8])
grid_size = parse(Float64, Base.split(data_name,"_")[10])

nnparams = get_custom_network_params("CNN_lidar", exp_name)

MIPVerify.setloglevel!("info")

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

#test_dataset = read_lidar_test_dataset("data/lidar_train_gaussian_0.001/valid.pickle")
#println("Fraction correct of validation set is $(frac_correct(nnparams, test_dataset, 2000))")

test_dataset_with_range = read_lidar_dataset_thread("data/$(data_name)/$(shape)_thread_$(thread_number).mat", shape)

MIPVerify.batch_verify_class_thread(
    nnparams,
    test_dataset_with_range,
    GurobiSolver(Gurobi.Env(), TimeLimit=1200),
    thread_number,
    save_path = joinpath("data", data_name),
    pp = MIPVerify.CustomPerturbationFamily(),
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=mip,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=1200),
)
