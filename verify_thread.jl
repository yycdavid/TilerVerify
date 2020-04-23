include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

# Argument: 1. exp_name 2. data_name 3. thread_number (4. time_limit)
exp_name = ARGS[1]
data_name = ARGS[2]
thread_number = parse(Int64, ARGS[3])

if length(ARGS) >= 4
    time_limit = parse(Float64, ARGS[4])
else
    time_limit = 1200
end

# Load model
nnparams = get_custom_network_params("CNN_small", exp_name)

MIPVerify.setloglevel!("info")

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

# test_dataset = read_custom_test_dataset("data/valid.mat")
# println("Average error of first 200 is $(average_error_across_labels(nnparams, test_dataset, 200))")

test_dataset_with_range = read_custom_dataset_thread("data/$(data_name)/thread_$(thread_number).mat")

MIPVerify.batch_find_error_bound_thread(
    nnparams,
    test_dataset_with_range,
    GurobiSolver(Gurobi.Env(), TimeLimit=time_limit),
    thread_number,
    save_path = joinpath("data", data_name),
    pp = MIPVerify.CustomPerturbationFamily(),
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=mip,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=time_limit),
)
