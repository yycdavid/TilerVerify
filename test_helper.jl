using MAT

function save_matrix_as_mat(data_matrix::Array{Float64,4})
    data_matrix = squeeze(data_matrix, 1)
    data_matrix = squeeze(data_matrix, 3)
    result = Dict()
    result["img"] = data_matrix
    matwrite("perturbed.mat", result)
end

function check_optimization_correctness(result_dict, nn)
    error = Dict()
    for setting in [:OffsetMin, :OffsetMax, :AngleMin, :AngleMax]
        output = result_dict[setting][:PerturbedInputValue] |> nn
        if setting == :OffsetMin || setting == :OffsetMax
            error[setting] = [result_dict[setting][:ObjectiveValue], output[1] - result_dict[setting][:ObjectiveValue]]
        else
            error[setting] = [result_dict[setting][:ObjectiveValue], output[2] - result_dict[setting][:ObjectiveValue]]
        end
    end
    return error
end
