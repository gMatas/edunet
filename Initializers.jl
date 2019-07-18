module Initializers

export
initialize,
Initializer,
HeUniform,
HeNormal,
RandomNormal,
RandomUniform,
Zeros,
Ones

include("Math.jl")

using Random

using .Math

abstract type Initializer end

function initialize(initializer::T where T<:Initializer); end

mutable struct HeNormal <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    HeNormal(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::HeNormal, rng::AbstractRNG = Random.GLOBAL_RNG) =
    Math.he_normal(
        initializer.type,
        initializer.dims,
        size(initializer.dims, 1),
        rng=rng)

mutable struct HeUniform <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    HeUniform(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::HeUniform, rng::AbstractRNG = Random.GLOBAL_RNG) =
    Math.he_uniform(
        initializer.type,
        initializer.dims,
        size(initializer.dims, 1),
        rng=rng)

mutable struct RandomUniform <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    RandomUniform(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::RandomUniform, rng::AbstractRNG = Random.GLOBAL_RNG) =
    Math.random_uniform(
        initializer.type,
        initializer.dims,
        rng=rng)

mutable struct RandomNormal <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    RandomNormal(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::RandomNormal, rng::AbstractRNG = Random.GLOBAL_RNG) =
    Math.random_normal(
        initializer.type,
        initializer.dims,
        rng=rng)

mutable struct Zeros <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    Zeros(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::Zeros, rng::Union{AbstractRNG, Nothing} = nothing) = zeros(
    initializer.type, initializer.dims)

mutable struct Ones <: Initializer
    type::Type{T} where T<:Number
    dims::Tuple{Vararg{Integer, N} where N}

    Ones(type::Type{T} where T<:Number,
        dims::Tuple{Vararg{Integer, N} where N}) = new(type, dims)
end

initialize(initializer::Ones, rng::Union{AbstractRNG, Nothing} = nothing) = ones(
    initializer.type, initializer.dims)

end
