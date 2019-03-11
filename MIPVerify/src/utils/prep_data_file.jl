const data_repo_path = "https://github.com/vtjeng/MIPVerify_data/raw/master"

function prep_data_file(relative_dir::String, filename::String)::String
    absolute_dir = joinpath(dependencies_path, relative_dir)
    if !ispath(absolute_dir)
        mkpath(absolute_dir)
    end
    
    relative_file_path = joinpath(relative_dir, filename)
    absolute_file_path = joinpath(dependencies_path, relative_file_path)
    if !isfile(absolute_file_path)
        url = joinpath(data_repo_path, relative_file_path)
        download(url, absolute_file_path)
    end

    return absolute_file_path
end