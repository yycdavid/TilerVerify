using CSV
using DataFrames
using MAT

data_name = ARGS[1]
num_threads = parse(Int64, ARGS[2])

offset_error_threshold = parse(Float64, ARGS[3])
angle_error_threshold = parse(Float64, ARGS[4])

main_path = joinpath("data", data_name)

# Collect summary files
thread_file_path = joinpath(main_path, "summary_0.csv")
summary_dt = CSV.read(thread_file_path)
for i in 1:(num_threads-1)
    thread_file_path = joinpath(main_path, "summary_$(i).csv")
    thread_dt = CSV.read(thread_file_path)
    summary_dt = vcat(summary_dt, thread_dt)
end

# Filter which one is solved and which needs solve again
not_solved = (summary_dt[:OffsetMinStatus] .!= "Optimal") .| (summary_dt[:OffsetMaxStatus] .!= "Optimal") .| (summary_dt[:AngleMinStatus] .!= "Optimal") .| (summary_dt[:AngleMaxStatus] .!= "Optimal")
offset_errors = max.(summary_dt[:OffsetMaxSolved] - summary_dt[:OffsetMin], summary_dt[:OffsetMax] - summary_dt[:OffsetMinSolved])
angle_errors = max.(summary_dt[:AngleMaxSolved] - summary_dt[:AngleMin], summary_dt[:AngleMax] - summary_dt[:AngleMinSolved])
solve_again = not_solved .| (offset_errors .> offset_error_threshold) .| (angle_errors .> angle_error_threshold)

solved_dt = summary_dt[.!solve_again, :]
solve_again_dt = summary_dt[solve_again, :]

# Save good results to a single file
overall_summary_path = joinpath(main_path, "summary.csv")
if isfile(overall_summary_path)
    rm(overall_summary_path)
end
CSV.write(overall_summary_path, solved_dt)

# Save bad results to another file
to_solve_path = joinpath(main_path, "to_solve.csv")
if isfile(to_solve_path)
    rm(to_solve_path)
end
CSV.write(to_solve_path, solve_again_dt)
