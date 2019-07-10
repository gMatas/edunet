module Math

export
random_normal,
random_uniform,
he_normal,
he_uniform,
relu,
relu_prime,
sigmoid,
sigmoid_prime,
softmax,
softmax_prime,
square_distance,
square_distance_prime,
cross_entropy,
cross_entropy_prime

using Random, Statistics, LinearAlgebra

function random_uniform(type, dims; minval=0, maxval=nothing, rng=Random.GLOBAL_RNG)
    if isnothing(maxval)
        maxval = type <: AbstractFloat ? 1.0 : typemax(type)
    else
        maxval = convert(type, maxval)
    end

    Random.rand(rng, type, dims...) * (maxval - minval) .+ minval
end

function random_normal(type, dims; mu=0, sigma=1.0, rng=Random.GLOBAL_RNG)
    y = Random.randn(rng, type, dims...)
    convert(Array{type}, y * sigma .- (mean(y) - mu))
    type.(y * sigma .- (mean(y) - mu))
end

function he_uniform(type, dims, n; minval=0, maxval=nothing, rng=Random.GLOBAL_RNG)
    y = random_uniform(type, dims, minval=minval, maxval=maxval, rng=rng)
    type.(y * sqrt(2 / n))
end

function he_normal(type, dims, n; mu=0, sigma=1.0, rng=Random.GLOBAL_RNG)
    y = random_normal(type, dims, mu=mu, sigma=sigma, rng=rng)
    type.(y * sqrt(2 / n))
end

function relu(x::Array{T, N} where {T <: AbstractFloat, N})
    y = copy(x)
    y[x .< 0] .= 0
    return y
end

function relu_prime(
        x::Array{T, N} where {T <: AbstractFloat, N},
        gradients::Array{T, N} where {T <: AbstractFloat, N}
    )
    indices = x .> 0
    dy = zeros(typeof(x), size(gradients))
    dy[indices] .= gradients[indices]
    return dy
end

function sigmoid(x::Array{T, N} where {T <: AbstractFloat, N})
    y = 1 ./ (1 + exp(-(x .- max(x...))))
    return y
end

function sigmoid_prime(
        x::Array{T, N} where {T <: AbstractFloat, N},
        gradients::Array{T, N} where {T <: AbstractFloat, N}
    )
    norm_x = x .- max(x...)
    dydx = exp(-norm_x) / ((1 .+ exp(-norm_x)) .^ 2)
    dx = gradients * dydx
    return dx
end

function softmax(x::Vector{T} where T <: AbstractFloat)
    exponents = exp(x .- max(x...))  # normalized exponents.
    y = exponents ./ sum(exponents)
    return y
end

function softmax_prime(
        x::Vector{T} where T <: AbstractFloat,
        gradients::Vector{T} where T <: AbstractFloat
    )
    y = softmax(x)
    dydx = diagm(0 => y) - (s * s')
    dx = gradients * dydx
    return dx
end

function square_distance(
        x::Vector{T} where T <: AbstractFloat,
        y::Vector{T} where T <: AbstractFloat
    )
    if size(x) != size(y)
        throw(DimensionMismatch("Input vectors dimensions does not match."))
    end

    e = (x - y) .^ 2
    return e
end

function square_distance_prime(
        x::Vector{T} where T <: AbstractFloat,
        y::Vector{T} where T <: AbstractFloat,
        gradients::Vector{T} where T <: AbstractFloat
    )
    if size(x) != size(y)
        throw(DimensionMismatch("Input vectors dimensions does not match."))
    end

    dydx = 2 .* (x - y)
    dx = gradients * dydx'
    return dx
end

function cross_entropy(
        x::Vector{T} where T <: AbstractFloat,
        y::Vector{T} where T <: AbstractFloat
    )
    if size(x) != size(y)
        throw(DimensionMismatch("Input vectors dimensions does not match."))
    end

    max_x = max(x...)
    norm_logsumexp = max_x + log(sum(exp(x .- max_x)))
    e = -dot(y, x .- norm_logsumexp)
    return e
end

function cross_entropy_prime(
        x::Vector{T} where T <: AbstractFloat,
        y::Vector{T} where T <: AbstractFloat,
        gradients::Vector{T} where T <: AbstractFloat
    )
    if size(x) != size(y)
        throw(DimensionMismatch("Input vectors dimensions does not match."))
    end

    dxdy = x - y
    dx = gradients * dxdy
    if size(dx, 1) > 1; return dx'; end
    return dx
end

end
