include("Utilities.jl")
include("Math.jl")
include("Layers.jl")
include("Initializers.jl")

using Images
using Random
using DSP

using Main.Utilities
using Main.Math
using Main.Layers
using Main.Initializers

function create_emoji_dataset_batch(image_1, image_2, n; ratio=0.5, rng=Random.GLOBAL_RNG)
    indexes = Random.randperm(rng, n)

    image_1 = Float32.(permutedims(channelview(smiley_image), (2, 3, 1)))
    image_2 = Float32.(permutedims(channelview(frown_image), (2, 3, 1)))

    first_half = Int(round(n * 0.5))
    second_half = n - first_half

    data_batch_1 = repeat(reshape(image_1, (1, size(image_1)...)), first_half, 1, 1, 1)
    data_batch_2 = repeat(reshape(image_2, (1, size(image_2)...)), second_half, 1, 1, 1)
    data_batch = cat(data_batch_1, data_batch_2, dims=1)

    labels_batch_1 = zeros(Float32, first_half, 2); labels_batch_1[:, 1] .= 1
    labels_batch_2 = zeros(Float32, second_half, 2); labels_batch_2[:, 2] .= 1
    labels_batch = cat(labels_batch_1, labels_batch_2, dims=1)

    return data_batch[indexes, :, :, :], labels_batch[indexes, :]
end


function main()

end


# ------------------------------------------------------------------------------

# Images
smiley_image_filepath = raw"/media/matthew/Data/My Work/Computational inteligence/mattcnn/god_damned_smile.bmp"
frown_image_filepath = raw"/media/matthew/Data/My Work/Computational inteligence/mattcnn/god_damned_frown.bmp"

smiley_image = Images.load(smiley_image_filepath)
frown_image = Images.load(frown_image_filepath)

# ------------------------------------------------------------------------------

n_epochs = 1
n_batches = 1
batch_size = 4

data_batch, labels_batch = create_emoji_dataset_batch(smiley_image, frown_image, batch_size)

i_sample = 1
sample_data = data_batch[i_sample, :, :, :]
sample_label = labels_batch[i_sample, :]

# ------------------------------------------------------------------------------

input_data_layer = Input(Float32, (20, 20, 3))
input_label_layer = Input(Float32, (2,))

conv_1 = Convolution2d(input_data_layer, 16, 3, mode="valid")
relu_1 = Relu(conv_1)

conv_2 = Convolution2d(relu_1, 16, 3, mode="valid")
relu_2 = Relu(conv_2)

conv_3 = Convolution2d(relu_2, 16, 3, mode="valid")
relu_3 = Relu(conv_3)

conv_4 = Convolution2d(relu_3, 16, 3, mode="valid")
relu_4 = Relu(conv_4)

conv_5 = Convolution2d(relu_4, 16, 3, mode="valid")
relu_5 = Relu(conv_5)

flat = Reshape(relu_5, (prod(relu_5.dims[1:2]), relu_5.dims[3]))





# for i_epoch = 1:n_epochs
#     for i_batch = 1:n_batches
#
#         data_batch, labels_batch = create_emoji_dataset_batch(smiley_image, frown_image, batch_size)
#         for i_sample = 1:length(data_batch)
#             sample_data = data_batch[i_sample, :, :, :]
#             sample_label = labels_batch[i_sample, :]
#
#
#             break
#         end
#     end
# end

main()
