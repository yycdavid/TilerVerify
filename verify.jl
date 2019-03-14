include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

nnparams = get_custom_network_params("CNN_small", "test_run")
test_dataset = read_custom_test_dataset("data/valid.mat")

MIPVerify.setloglevel!("info")

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

# println("Average error of first 200 is $(average_error_across_labels(nnparams, test_dataset, 200))")

test_dataset_with_range = read_custom_dataset_with_range("data/test_verify.mat")

input_lower_bound = test_dataset_with_range.image_lower_bounds[1:1,:,:,:]
input_upper_bound = test_dataset_with_range.image_upper_bounds[1:1,:,:,:]
offset_low = test_dataset_with_range.offset_lower_bounds[1]
offset_high = test_dataset_with_range.offset_upper_bounds[1]
angle_low = test_dataset_with_range.angle_lower_bounds[1]
angle_high = test_dataset_with_range.angle_upper_bounds[1]


result_dict = MIPVerify.find_range_for_outputs(
    nnparams,
    input_lower_bound,
    input_upper_bound,
    GurobiSolver(Gurobi.Env(), TimeLimit=1200),
    pp = MIPVerify.CustomPerturbationFamily(),
    tightening_algorithm=mip,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=20),
)


"""
MIPVerify.batch_find_untargeted_attack(
    nnparams,
    mnist.test,
    1:10000,
    GurobiSolver(Gurobi.Env(), BestObjStop=0.1, TimeLimit=1200),
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
    norm_order=Inf,
    rebuild=true,
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=lp,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=20),
    cache_model=false,
    solve_if_predicted_in_targeted=false,
)
"""
