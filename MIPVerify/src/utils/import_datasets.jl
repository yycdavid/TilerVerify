using MAT

export read_datasets, read_custom_test_dataset

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
