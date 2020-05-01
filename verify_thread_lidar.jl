include("./MIPVerify/src/MIPVerify.jl")
using MIPVerify
using Gurobi

# Argument: 1. exp_name 2. data_name 3. thread_number
exp_name = ARGS[1]
data_name = ARGS[2]
shape = parse(Int64, ARGS[3])
thread_number = parse(Int64, ARGS[4])
save_dir = ARGS[5]

if length(ARGS) >= 6
    time_limit = parse(Float64, ARGS[6])
    time_limit_tightening = 0.1
else
    time_limit = 1200
    time_limit_tightening = 1200
end

# primarily meant as a sanity check.
# note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

#test_dataset = read_lidar_test_dataset("data/lidar_train_gaussian_0.001/valid.pickle")
#println("Fraction correct of validation set is $(frac_correct(nnparams, test_dataset, 2000))")

if isfile("data/$(data_name)/$(shape)_thread_$(thread_number).mat")
    # Load model
    nnparams = get_custom_network_params("CNN_lidar", exp_name)

    MIPVerify.setloglevel!("info")


    # primarily meant as a sanity check.
    # note that determining the fraction correct for is highly inefficient, and can be very slow for large networks!

    # test_dataset = read_custom_test_dataset("data/valid.mat")
    # println("Average error of first 200 is $(average_error_across_labels(nnparams, test_dataset, 200))")

    test_dataset_with_range = read_lidar_dataset_thread("data/$(data_name)/$(shape)_thread_$(thread_number).mat", shape)

    if save_dir == ""
        save_name = data_name
    else
        save_name = joinpath(data_name, save_dir)
    end

    if !isdir(joinpath("data", save_name))
        mkdir(joinpath("data", save_name))
    end

    MIPVerify.batch_verify_class_thread(
        nnparams,
        test_dataset_with_range,
        GurobiSolver(Gurobi.Env(), TimeLimit=time_limit),
        thread_number,
        shape,
        save_path = joinpath("data", save_name),
        pp = MIPVerify.CustomPerturbationFamily(),
        solve_rerun_option = MIPVerify.never,
        tightening_algorithm=mip,
        tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=time_limit_tightening),
    )

else
    println("Shape $(shape), Thread $(thread_number) has no data\n")
end
