using CSV
using DataFrames
using MAT

data_name = ARGS[1]
num_threads = parse(Int64, ARGS[2])

VERIFIED_TRUE = 0
VERIFIED_FALSE = 1
NOT_SURE = 2

verify_result = Dict()

main_path = joinpath("data", data_name)
for label in 0:2
    # Collect summary files
    thread_file_path = joinpath(main_path, "$(label)_summary_0.csv")
    summary_dt = CSV.read(thread_file_path)
    for i in 1:(num_threads-1)
        thread_file_path = joinpath(main_path, "$(label)_summary_$(i).csv")
        thread_dt = CSV.read(thread_file_path)
        summary_dt = vcat(summary_dt, thread_dt)
    end
    # Save to a single file
    overall_summary_path = joinpath(main_path, "$(label)_summary.csv")
    if isfile(overall_summary_path)
        rm(overall_summary_path)
    end
    CSV.write(overall_summary_path, summary_dt)
    verified = summary_dt[:Verified]
    solve_status_1 = summary_dt[:LogitDiff1Status]
    solve_status_2 = summary_dt[:LogitDiff2Status]
    verify_result["VerifyStatus_$(label)"] = Int64[]
    for i in 1:length(verified)
        if (verified[i]==false)
            push!(verify_result["VerifyStatus_$(label)"], VERIFIED_FALSE)
        elseif (solve_status_1[i] != "Optimal") || (solve_status_2[i] != "Optimal")
            push!(verify_result["VerifyStatus_$(label)"], NOT_SURE)
        else
            push!(verify_result["VerifyStatus_$(label)"], VERIFIED_TRUE)
        end
    end
end

info_path = joinpath(main_path, "info.mat")
info = matread(info_path)
verify_result["distance_grid_num"] = info["distance_grid_num"]
verify_result["angle_grid_num"] = info["angle_grid_num"]
matwrite(joinpath(main_path, "verify_result.mat"), verify_result)
