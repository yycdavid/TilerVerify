using MAT

export read_datasets, read_custom_test_dataset, read_custom_dataset_with_range, read_custom_dataset_thread

abstract type Dataset end

abstract type LabelledDataset<:Dataset end

"""
$(TYPEDEF)

Dataset of images stored as a 4-dimensional array of size `(num_samples, image_height,
image_width, num_channels)`, with accompanying labels (sorted in the same order) of size
`num_samples`.
"""
struct LabelledImageDataset{T<:Real, U<:Integer} <: LabelledDataset
    images::Array{T, 4}
    labels::Array{U, 1}

    function LabelledImageDataset{T, U}(images::Array{T, 4}, labels::Array{U, 1})::LabelledImageDataset where {T<:Real, U<:Integer}
        (num_image_samples, image_height, image_width, num_channels) = size(images)
        (num_label_samples, ) = size(labels)
        @assert num_image_samples==num_label_samples
        return new(images, labels)
    end
end

function LabelledImageDataset(images::Array{T, 4}, labels::Array{U, 1})::LabelledImageDataset where {T<:Real, U<:Integer}
    LabelledImageDataset{T, U}(images, labels)
end

function num_samples(dataset::LabelledDataset)
    return length(dataset.labels)
end

function Base.show(io::IO, dataset::LabelledImageDataset)
    image_size = size(dataset.images[1, :, :, :])
    num_samples = MIPVerify.num_samples(dataset)
    min_pixel = minimum(dataset.images)
    max_pixel = maximum(dataset.images)
    min_label = minimum(dataset.labels)
    max_label = maximum(dataset.labels)
    num_unique_labels = length(unique(dataset.labels))
    print(io,
        "{LabelledImageDataset}",
        "\n    `images`: $num_samples images of size $image_size, with pixels in [$min_pixel, $max_pixel].",
        "\n    `labels`: $num_samples corresponding labels, with $num_unique_labels unique labels in [$min_label, $max_label]."
    )
end

# DoubleLabeledImageDataset is the testset for checking ordinary performance of the model

struct DoubleLabeledImageDataset{T<:Real, U<:Real, V<:Real}
    images::Array{T, 4}
    offsets::Array{U, 1}
    angles::Array{V, 1}

    function DoubleLabeledImageDataset{T, U, V}(images::Array{T, 4}, offsets::Array{U, 1}, angles::Array{V, 1})::DoubleLabeledImageDataset where {T<:Real, U<:Real, V<:Real}
        (num_image_samples, image_height, image_width, num_channels) = size(images)
        (num_offset_samples, ) = size(offsets)
        (num_angle_samples, ) = size(angles)
        @assert num_image_samples==num_offset_samples
        @assert num_offset_samples==num_angle_samples
        return new(images, offsets, angles)
    end
end

function DoubleLabeledImageDataset(images::Array{T, 4}, offsets::Array{U, 1}, angles::Array{V, 1})::DoubleLabeledImageDataset where {T<:Real, U<:Real, V<:Real}
    DoubleLabeledImageDataset{T, U, V}(images, offsets, angles)
end

function num_samples(dataset::DoubleLabeledImageDataset)
    return length(dataset.offsets)
end

function Base.show(io::IO, dataset::DoubleLabeledImageDataset)
    image_size = size(dataset.images[1, :, :, :])
    num_samples = MIPVerify.num_samples(dataset)
    min_pixel = minimum(dataset.images)
    max_pixel = maximum(dataset.images)
    min_offset = minimum(dataset.offsets)
    max_offset = maximum(dataset.offsets)
    min_angle = minimum(dataset.angles)
    max_angle = maximum(dataset.angles)
    print(io,
        "{DoubleLabeledImageDataset}",
        "\n    `images`: $num_samples images of size $image_size, with pixels in [$min_pixel, $max_pixel].",
        "\n    `offsets, angles`: $num_samples corresponding labels, offsets in [$min_offset, $max_offset] and angles in [$min_angle, $max_angle]."
    )
end


# RangeDataset is the testset for computing maximum error

struct RangeDataset{T<:Real, U<:Real, V<:Real}
    image_lower_bounds::Array{T, 4}
    image_upper_bounds::Array{T, 4}
    offset_lower_bounds::Array{U, 1}
    offset_upper_bounds::Array{U, 1}
    angle_lower_bounds::Array{V, 1}
    angle_upper_bounds::Array{V, 1}
    images::Array{T, 4}
    offsets::Array{U, 1}
    angles::Array{V, 1}
    offset_grid_num::Integer
    angle_grid_num::Integer
    grid_size::Real

    function RangeDataset{T, U, V}(
        image_lower_bounds::Array{T, 4},
        image_upper_bounds::Array{T, 4},
        offset_lower_bounds::Array{U, 1},
        offset_upper_bounds::Array{U, 1},
        angle_lower_bounds::Array{V, 1},
        angle_upper_bounds::Array{V, 1},
        images::Array{T, 4},
        offsets::Array{U, 1},
        angles::Array{V, 1},
        offset_grid_num::Integer,
        angle_grid_num::Integer,
        grid_size::Real
        )::RangeDataset where {T<:Real, U<:Real, V<:Real}
        return new(image_lower_bounds, image_upper_bounds, offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds, images, offsets, angles, offset_grid_num, angle_grid_num, grid_size)
    end
end

function RangeDataset(
    image_lower_bounds::Array{T, 4},
    image_upper_bounds::Array{T, 4},
    offset_lower_bounds::Array{U, 1},
    offset_upper_bounds::Array{U, 1},
    angle_lower_bounds::Array{V, 1},
    angle_upper_bounds::Array{V, 1},
    images::Array{T, 4},
    offsets::Array{U, 1},
    angles::Array{V, 1},
    offset_grid_num::Integer,
    angle_grid_num::Integer,
    grid_size::Real
    )::RangeDataset where {T<:Real, U<:Real, V<:Real}
    RangeDataset{T, U, V}(image_lower_bounds, image_upper_bounds, offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds, images, offsets, angles, offset_grid_num, angle_grid_num, grid_size)
end

function num_samples(dataset::RangeDataset)
    return length(dataset.offset_lower_bounds)
end

function Base.show(io::IO, dataset::RangeDataset)
    image_size = size(dataset.image_lower_bounds[1, :, :, :])
    num_samples = MIPVerify.num_samples(dataset)
    min_offset = minimum(dataset.offset_lower_bounds)
    max_offset = maximum(dataset.offset_upper_bounds)
    min_angle = minimum(dataset.angle_lower_bounds)
    max_angle = maximum(dataset.angle_upper_bounds)
    print(io,
        "{RangeDataset}",
        "\n    `image_lower_bounds, image_upper_bounds`: $num_samples test points, image size $image_size.",
        "\n    `offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds`: offsets in [$min_offset, $max_offset] and angles in [$min_angle, $max_angle]."
    )
end



struct RangeThreadDataset{T<:Real, U<:Real, V<:Real}
    image_lower_bounds::Array{T, 4}
    image_upper_bounds::Array{T, 4}
    offset_lower_bounds::Array{U, 1}
    offset_upper_bounds::Array{U, 1}
    angle_lower_bounds::Array{V, 1}
    angle_upper_bounds::Array{V, 1}
    images::Array{T, 4}
    offsets::Array{U, 1}
    angles::Array{V, 1}
    index::Array{<:Integer, 1}

    function RangeThreadDataset{T, U, V}(
        image_lower_bounds::Array{T, 4},
        image_upper_bounds::Array{T, 4},
        offset_lower_bounds::Array{U, 1},
        offset_upper_bounds::Array{U, 1},
        angle_lower_bounds::Array{V, 1},
        angle_upper_bounds::Array{V, 1},
        images::Array{T, 4},
        offsets::Array{U, 1},
        angles::Array{V, 1},
        index::Array{<:Integer, 1}
        )::RangeThreadDataset where {T<:Real, U<:Real, V<:Real}
        return new(image_lower_bounds, image_upper_bounds, offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds, images, offsets, angles, index)
    end
end

function RangeThreadDataset(
    image_lower_bounds::Array{T, 4},
    image_upper_bounds::Array{T, 4},
    offset_lower_bounds::Array{U, 1},
    offset_upper_bounds::Array{U, 1},
    angle_lower_bounds::Array{V, 1},
    angle_upper_bounds::Array{V, 1},
    images::Array{T, 4},
    offsets::Array{U, 1},
    angles::Array{V, 1},
    index::Array{<:Integer, 1}
    )::RangeThreadDataset where {T<:Real, U<:Real, V<:Real}
    RangeThreadDataset{T, U, V}(image_lower_bounds, image_upper_bounds, offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds, images, offsets, angles, index)
end

function num_samples(dataset::RangeThreadDataset)
    return length(dataset.offset_lower_bounds)
end

function Base.show(io::IO, dataset::RangeThreadDataset)
    image_size = size(dataset.image_lower_bounds[1, :, :, :])
    num_samples = MIPVerify.num_samples(dataset)
    min_offset = minimum(dataset.offset_lower_bounds)
    max_offset = maximum(dataset.offset_upper_bounds)
    min_angle = minimum(dataset.angle_lower_bounds)
    max_angle = maximum(dataset.angle_upper_bounds)
    print(io,
        "{RangeThreadDataset}",
        "\n    `image_lower_bounds, image_upper_bounds`: $num_samples test points, image size $image_size.",
        "\n    `offset_lower_bounds, offset_upper_bounds, angle_lower_bounds, angle_upper_bounds`: offsets in [$min_offset, $max_offset] and angles in [$min_angle, $max_angle]."
    )
end


"""
$(TYPEDEF)

Named dataset containing a training set and a test set which are expected to contain the
same kind of data.

$(FIELDS)
"""
struct NamedTrainTestDataset{T<:Dataset, U<:Dataset} <: Dataset
    """
    Name of dataset.
    """
    name::String
    """
    Training set.
    """
    train::T
    """
    Test set.
    """
    test::U
    # TODO (vtjeng): train and test should be the same type of struct (but might potentially have different parameters).
end

function Base.show(io::IO, dataset::NamedTrainTestDataset)
    print(io,
        "$(dataset.name):",
        "\n  `train`: $(dataset.train)",
        "\n  `test`: $(dataset.test)"
    )
end

"""
$(SIGNATURES)

Makes popular machine learning datasets available as a `NamedTrainTestDataset`.

# Arguments
* `name::String`: name of machine learning dataset. Options:
    * `MNIST`: [The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/). Pixel values in original dataset are provided as uint8 (0 to 255), but are scaled to range from 0 to 1 here.
    * `CIFAR10`: [Labelled subset in 10 classes of 80 million tiny images dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Pixel values in original dataset are provided as uint8 (0 to 255), but are scaled to range from 0 to 1 here.
"""
function read_datasets(name::String)::NamedTrainTestDataset
    name = lowercase(name)

    if name in ["mnist", "cifar10"]
        dir = joinpath("datasets", name)

        m_train = prep_data_file(dir, "$(name)_int_train.mat") |> matread
        train = LabelledImageDataset(m_train["images"]/255, m_train["labels"][:])

        m_test = prep_data_file(dir, "$(name)_int_test.mat") |> matread
        test = LabelledImageDataset(m_test["images"]/255, m_test["labels"][:])
        return NamedTrainTestDataset(name, train, test)
    else
        throw(DomainError("Dataset $name not supported."))
    end
end

function read_custom_test_dataset(relative_path::String)::DoubleLabeledImageDataset
    # Argument is the relative path of the .mat file to the root directory of the repo
    # Stored data images does not have the channel dimension. So here add it
    absolute_path = joinpath(root_path, relative_path)
    test_data = matread(absolute_path)
    test_data["images"] = reshape(test_data["images"], (size(test_data["images"])...,1))
    return DoubleLabeledImageDataset(test_data["images"]/255, test_data["offsets"][:], test_data["angles"][:])
end

function read_custom_dataset_with_range(relative_path::String)::RangeDataset
    absolute_path = joinpath(root_path, relative_path)
    test_data = matread(absolute_path)
    # Get grid size
    grid_size = parse(Float64, Base.split(relative_path,"_")[end][1:end-4])
    # Add channel dimension
    test_data["image_lower_bounds"] = reshape(test_data["image_lower_bounds"], (size(test_data["image_lower_bounds"])...,1)) #(N,H,W,C)
    test_data["image_upper_bounds"] = reshape(test_data["image_upper_bounds"], (size(test_data["image_upper_bounds"])...,1)) #(N,H,W,C)
    test_data["images"] = reshape(test_data["images"], (size(test_data["images"])...,1)) #(N,H,W,C)
    return RangeDataset(test_data["image_lower_bounds"]/255, test_data["image_upper_bounds"]/255, test_data["offset_lower_bounds"][:], test_data["offset_upper_bounds"][:], test_data["angle_lower_bounds"][:], test_data["angle_upper_bounds"][:], test_data["images"], test_data["offsets"][:], test_data["angles"][:], test_data["offset_grid_num"], test_data["angle_grid_num"], grid_size)
end

function read_custom_dataset_thread(relative_path::String)::RangeThreadDataset
    absolute_path = joinpath(root_path, relative_path)
    test_data = matread(absolute_path)
    # Add channel dimension
    test_data["image_lower_bounds"] = reshape(test_data["image_lower_bounds"], (size(test_data["image_lower_bounds"])...,1)) #(N,H,W,C)
    test_data["image_upper_bounds"] = reshape(test_data["image_upper_bounds"], (size(test_data["image_upper_bounds"])...,1)) #(N,H,W,C)
    test_data["images"] = reshape(test_data["images"], (size(test_data["images"])...,1)) #(N,H,W,C)
    return RangeThreadDataset(test_data["image_lower_bounds"]/255, test_data["image_upper_bounds"]/255, test_data["offset_lower_bounds"][:], test_data["offset_upper_bounds"][:], test_data["angle_lower_bounds"][:], test_data["angle_upper_bounds"][:], test_data["images"], test_data["offsets"][:], test_data["angles"][:], test_data["index"][:])
end
