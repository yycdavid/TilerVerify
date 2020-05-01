using CSV
using DataFrames
using MAT

data_name = ARGS[1]
num_threads = parse(Int64, ARGS[2])
save_folder = ARGS[3]

VERIFIED_TRUE = 0
VERIFIED_FALSE = 1
NOT_SURE = 2

verify_result = Dict()

exp_path = joinpath("data", data_name)
if save_folder == ""
    main_path = exp_path
else
    main_path = joinpath(exp_path, save_folder)
end
for label in 0:2
    # Collect summary files
    thread_file_path = joinpath(main_path, "$(label)_summary_0.csv")
    if isfile(thread_file_path)
        summary_dt = CSV.read(thread_file_path)
        for i in 1:(num_threads-1)
            thread_file_path = joinpath(main_path, "$(label)_summary_$(i).csv")
            if isfile(thread_file_path)
                thread_dt = CSV.read(thread_file_path)
                summary_dt = vcat(summary_dt, thread_dt)
            end
        end

        # Filter which one is solved and which needs solve again
        verified = summary_dt[:Verified]
        solve_status_1 = summary_dt[:LogitDiff1Status]
        solve_status_2 = summary_dt[:LogitDiff2Status]

        solve_again = (verified .== false) .| (solve_status_1 .!= "Optimal") .| (solve_status_2 .!= "Optimal")

        solved_dt = summary_dt[.!solve_again, :]
        solve_again_dt = summary_dt[solve_again, :]

        # Save to a single file
        overall_summary_path = joinpath(main_path, "$(label)_summary.csv")
        if isfile(overall_summary_path)
            rm(overall_summary_path)
        end
        CSV.write(overall_summary_path, solved_dt)


        # Save bad results to another file
        if nrow(solve_again_dt) > 0
            to_solve_path = joinpath(main_path, "$(label)_to_solve.csv")
            if isfile(to_solve_path)
                rm(to_solve_path)
            end
            CSV.write(to_solve_path, solve_again_dt)
        end
    end
end
