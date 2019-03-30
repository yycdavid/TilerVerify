using CSV
using DataFrames
using MAT

data_name = ARGS[1]
num_threads = parse(Int64, ARGS[2])

main_path = joinpath("data", data_name)
# Collect summary files
thread_file_path = joinpath(main_path, "summary_0.csv")
summary_dt = CSV.read(thread_file_path)
for i in 1:(num_threads-1)
    thread_file_path = joinpath(main_path, "summary_$(i).csv")
    thread_dt = CSV.read(thread_file_path)
    summary_dt = vcat(summary_dt, thread_dt)
end
# Save to a single file
overall_summary_path = joinpath(main_path, "summary.csv")
if isfile(overall_summary_path)
    rm(overall_summary_path)
end
CSV.write(overall_summary_path, summary_dt)


# Get error matrix
offset_errors = max.(summary_dt[:OffsetMaxSolved] - summary_dt[:OffsetMin], summary_dt[:OffsetMax] - summary_dt[:OffsetMinSolved])
angle_errors = max.(summary_dt[:AngleMaxSolved] - summary_dt[:AngleMin], summary_dt[:AngleMax] - summary_dt[:AngleMinSolved])
error_result = Dict()
error_result["offset_errors"] = offset_errors
error_result["angle_errors"] = angle_errors

info_path = joinpath(main_path, "info.mat")
info = matread(info_path)
error_result["offset_grid_num"] = info["offset_grid_num"]
error_result["angle_grid_num"] = info["angle_grid_num"]
matwrite(joinpath(main_path, "error_bound_result.mat"), error_result)
