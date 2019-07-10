module Layers

export
feed!,
forward!,
backward!,
Input,
Convolution2d,
Reshape,
Relu

include("Math.jl")
include("Utilities.jl")
include("Initializers.jl")

using Random

using .Math
using .Initializers
import .Utilities: convolution2d

abstract type LayerCache end

function clear_cache!(cache::T where T<:LayerCache)
    cache.outputs = nothing
end

macro implement_layer_cache(name::String, outputs_type::Expr)
    struct_name = Symbol(name)
    return quote
        mutable struct $(struct_name) <: LayerCache
            outputs::Union{$outputs_type, Nothing}
            $(struct_name)() = new(nothing)
        end
    end
end

abstract type Layer end

get_inputs(layer::Layer) = layer.input_layer.cache.outputs

get_outputs(layer::Layer) = layer.cache.outputs

function set_outputs!(layer::Layer, outputs); layer.cache.outputs = outputs; end

clear_cache!(layer::T where T<:Layer) = clear_cache!(level.cache)

@implement_layer_cache("InputCache", Array{T, N} where {T<:AbstractFloat, N})

mutable struct Input <: Layer
    value::Array{T, N} where {T<:AbstractFloat, N}
    type::Type{<:AbstractFloat}
    dims::Tuple{Vararg{Integer, N} where N}
    cache::InputCache

    function Input(type::Type{<:AbstractFloat} , dims::Tuple{Vararg{Integer, N} where N})
        cache = InputCache()
        cache.outputs = Array{type, length(dims)}(undef, dims)
        new(cache.outputs, type, dims, cache)
    end
end

function feed!(layer::Input, value::Array{T, N} where {T<:AbstractFloat, N})
    layer.value = value
end

function feed!(feed_dict::Dict{Input, Array{T, N} where {T<:AbstractFloat, N}})
    for (layer, value) in pairs(feed_dict)
        layer.value = value
    end
end

function feed!(feed_pairs::Pair{<:Input, <:Array{<:AbstractFloat, N} where N}...)
    for (layer, value) in feed_pairs
        layer.value = value
    end
end

function forward!(layer::Input)
    outputs = layer.value
    set_outputs!(layer, outputs)
    return outputs
end

function backward!(layer::Input) end

@implement_layer_cache("Convolution2dCache", Array{<:AbstractFloat, 3})

mutable struct Convolution2d <: Layer
    input_layer::T where T<:Layer
    weights::Array{T, 4} where T<:AbstractFloat
    bias::Array{T, 2} where T<:AbstractFloat
    x_stride::Integer
    y_stride::Integer
    mode::String
    trainable::Bool
    type::Type{<:AbstractFloat}
    dims::Tuple{Vararg{Integer, N} where N}
    cache::Convolution2dCache

    function Convolution2d(
            input_layer::T where T<:Layer,
            filters::Integer,
            kernel_size::Union{Integer, Tuple{Integer, Integer}};
            strides::Union{Integer, Tuple{Integer, Integer}} = (1, 1),
            weights_initializer::Type{T} where T<:Initializer = Initializers.HeUniform,
            bias_initializer::Type{T} where T<:Initializer = Initializers.Zeros,
            mode::String = "valid",
            trainable::Bool = true,
            rng::AbstractRNG = Random.GLOBAL_RNG
        )
        input_type = input_layer.type
        input_size = input_layer.dims

        # Initialize kernel and bias tensors.
        y_kernel_size, x_kernel_size = isa(kernel_size, Integer) ?
            (kernel_size, kernel_size) : kernel_size
        kernel_dims = (filters, y_kernel_size, x_kernel_size, input_size[3])

        bias_dims = (filters, 1)
        weights = initialize(weights_initializer(input_type, kernel_dims), rng)
        bias = initialize(bias_initializer(input_type, bias_dims), rng)

        # Set stride parameters.
        y_stride, x_stride = isa(strides, Integer) ? (strides, strides) : strides

        # Pre-compute layer output size.
        mode = lowercase(mode)
        output_size_y, output_size_x = Utilities.compute_convolution_output_size_2d(
            input_size[1:2], (y_kernel_size, x_kernel_size), (y_stride, x_stride), mode)
        output_size = (output_size_y, output_size_x, filters)

        new(input_layer, weights, bias, x_stride, y_stride, mode, trainable,
            input_type, output_size, Convolution2dCache())
    end
end

function forward!(layer::Convolution2d)
    inputs = get_inputs(layer)
    type = eltype(inputs)
    weights = type.(layer.weights)
    n_filters = size(weights, 1)

    conv_out = convolution2d(inputs, weights[1,:,:,:],
        x_stride=layer.x_stride, y_stride=layer.y_stride, mode=layer.mode)

    if n_filters == 1; return conv_out; end

    out = Array{type, 3}(undef, (size(conv_out)[1:2]..., n_filters))
    out[:, :, 1] = conv_out

    for i = 2:n_filters
        conv_out = convolution2d(inputs, weights[1,:,:,:],
            x_stride=layer.x_stride, y_stride=layer.y_stride, mode=layer.mode)
        out[:, :, i] = conv_out
    end

    set_outputs!(layer, out)
    return out
end

function backward!(layer::Convolution2d)
end

@implement_layer_cache("ReshapeCache", Array{<:AbstractFloat, N} where N)

mutable struct Reshape <: Layer
    input_layer::T where T<:Layer

    dims::Tuple{Vararg{Integer, N} where N}
    cache::ReshapeCache

    Reshape(input_layer::T where T<:Layer, dims::Tuple{Vararg{Integer, N} where N}) = new(
        input_layer, dims, ReshapeCache())
end

function forward!(layer::Reshape)
    inputs = get_inputs(layer)
    outputs = reshape(inputs, layer.dims)
    set_outputs!(layer, outputs)
    return outputs
end

function backward!(layer::Reshape)
    outputs = reshape(layer.cache.outputs, size(layer.cache.inputs))
    return outputs
end

@implement_layer_cache("ReluCache", Array{T, N} where {T <: AbstractFloat, N})

mutable struct Relu <: Layer
    input_layer::T where T<:Layer
    type::Type{<:AbstractFloat}
    dims::Tuple{Vararg{Integer, N} where N}
    cache::ReluCache
    Relu(input_layer::T where T<:Layer) = new(
        input_layer, input_layer.type, input_layer.dims, ReluCache())
end

function forward!(layer::Relu)
    inputs = get_inputs(layer)
    outputs = Math.relu(inputs)
    set_outputs!(layer, outputs)
    return outputs
end

function backward!(layer::Relu)
end

# TODO: Implement Dense layer.

# @implement_layer_cache("DenseCache")
#
# mutable struct Dense <: Layer
#     input_layer::T where T<:Layer
#     cache::DenseCache
#
#     Dense(input_layer::T where T<:Layer) = new(input_layer, DenseCache())
# end
#
# function forward!(layer::Dense)
#     inputs = get_cached_inputs(layer.input_layer)
#     update_cache(layer)
# end
#
# function backward!(layer::Dense)
# end

end
