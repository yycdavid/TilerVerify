include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

exp_name = "big_100000"

MIPVerify.setloglevel!("info")

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

# test_dataset = read_custom_test_dataset("data/valid.mat")
# println("Average error of first 200 is $(average_error_across_labels(nnparams, test_dataset, 200))")

test_dataset_with_range = read_custom_dataset_with_range("data/test_verify_$(ARGS[1])_$(ARGS[2]).mat")

num_threads = Threads.nthreads()
println("Number of threads available is $(num_threads)")
MIPVerify.batch_find_error_bound_parallel(
    [get_custom_network_params("CNN_small", exp_name) for i in 1:num_threads],
    test_dataset_with_range,
    [GurobiSolver(Gurobi.Env(), TimeLimit=1200) for i in 1:num_threads],
    [GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=1200) for i in 1:num_threads],
    save_path = joinpath("trained_models", exp_name),
    pp = MIPVerify.CustomPerturbationFamily(),
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=mip,
)

#=
time_spent = @elapsed MIPVerify.batch_find_error_bound(
    get_custom_network_params("CNN_small", exp_name),
    test_dataset_with_range,
    GurobiSolver(Gurobi.Env(), TimeLimit=1200),
    save_path = joinpath("trained_models", exp_name),
    pp = MIPVerify.CustomPerturbationFamily(),
    solve_rerun_option = MIPVerify.never,
    tightening_algorithm=mip,
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=1200),
)
=#

#=
include("./test_helper.jl")

summary = Any[]
for i in 1:2
    input_lower_bound = test_dataset_with_range.image_lower_bounds[i:i,:,:,:]
    input_upper_bound = test_dataset_with_range.image_upper_bounds[i:i,:,:,:]
    offset_low = test_dataset_with_range.offset_lower_bounds[i]
    offset_high = test_dataset_with_range.offset_upper_bounds[i]
    angle_low = test_dataset_with_range.angle_lower_bounds[i]
    angle_high = test_dataset_with_range.angle_upper_bounds[i]
    summary_item = Dict()
    summary_item[:offset] = [offset_low, offset_high]
    summary_item[:angle] = [angle_low, angle_high]

    result_dict = MIPVerify.find_range_for_outputs(
        nnparams,
        input_lower_bound,
        input_upper_bound,
        GurobiSolver(Gurobi.Env(), TimeLimit=1200),
        pp = MIPVerify.CustomPerturbationFamily(),
        tightening_algorithm=mip,
        tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=20),
    )
    summary_item[:optimized] = check_optimization_correctness(result_dict, nnparams)
    summary_item[:time] = result_dict[:TotalTime]
    append!(summary, summary_item)
end

#save_matrix_as_mat(result_dict[:OffsetMin][:PerturbedInputValue])
=#

#=
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
=#
