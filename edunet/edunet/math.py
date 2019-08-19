from typing import Sequence, Iterable, Union, Tuple
import math

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState


def random_uniform(
        shape: Sequence[int],
        minval: Union[int, float] = 0,
        maxval: Union[None, int, float] = None,
        dtype: Union[type, np.dtype] = float,
        random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None
):
    rand = random_state if isinstance(random_state, RandomState) else np.random.RandomState(random_state)

    dtype = np.dtype(dtype)
    is_float = dtype.kind in {'f'}
    is_integer = dtype.kind in {'u', 'i'}

    if is_float:
        if maxval is None:
            maxval = 1.0
        y = rand.random_sample(shape) * (maxval - minval) + minval
    elif is_integer:
        if maxval is None:
            maxval = np.iinfo(dtype).max
        y = rand.randint(int(minval), maxval, shape)
    else:
        raise TypeError('`dtype` must be numerical real type.')

    return np.array(y, dtype)


#function random_uniform(type, dims; minval=0, maxval=nothing, rng=Random.GLOBAL_RNG)
#    if isnothing(maxval)
#        maxval = type <: AbstractFloat ? 1.0 : typemax(type)
#    else
#        maxval = convert(type, maxval)
#    end

#    Random.rand(rng, type, dims...) * (maxval - minval) .+ minval
#end


def random_normal(
        shape: Sequence[int],
        mu: Union[int, float] = 0,
        sigma: Union[int, float] = 1.0,
        dtype: Union[type, np.dtype] = float,
        random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None
):
    rand = random_state if isinstance(random_state, RandomState) else np.random.RandomState(random_state)
    y = rand.standard_normal(shape)
    y = y * sigma - (y.mean() - mu)

    dtype = np.dtype(dtype)
    return np.array(y, dtype)


#function random_normal(type, dims; mu=0, sigma=1.0, rng=Random.GLOBAL_RNG)
#    y = Random.randn(rng, type, dims...)
#    convert(Array{type}, y * sigma .- (mean(y) - mu))
#    type.(y * sigma .- (mean(y) - mu))
#end


def he_uniform(
        shape: Sequence[int],
        n: int,
        minval: Union[int, float] = 0,
        maxval: Union[None, int, float] = None,
        dtype: Union[type, np.dtype] = float,
        random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None
):
    y = random_uniform(shape, minval, maxval, dtype, random_state)
    y *= math.sqrt(2.0 / n)
    return y


#function he_uniform(type, dims, n; minval=0, maxval=nothing, rng=Random.GLOBAL_RNG)
#    y = random_uniform(type, dims, minval=minval, maxval=maxval, rng=rng)
#    type.(y * sqrt(2 / n))
#end


def he_normal(
        shape: Sequence[int],
        n: int,
        mu: Union[int, float] = 0,
        sigma: Union[int, float] = 1.0,
        dtype: Union[type, np.dtype] = float,
        random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None
):
    y = random_normal(shape, mu, sigma, dtype, random_state)
    y *= math.sqrt(2.0 / n)
    return y


#function he_normal(type, dims, n; mu=0, sigma=1.0, rng=Random.GLOBAL_RNG)
#    y = random_normal(type, dims, mu=mu, sigma=sigma, rng=rng)
#    type.(y * sqrt(2 / n))
#end


def relu(x: ndarray):
    y = x.copy()
    y[x < 0] = 0
    return y


#function relu(x::Array{T, N} where {T <: AbstractFloat, N})
#    y = copy(x)
#    y[x .< 0] .= 0
#    return y
#end


def relu_prime(x: ndarray, gradients: ndarray):
    indices = x > 0
    dy = np.zeros(gradients.size, gradients.dtype)
    dy[indices] = gradients[indices]
    return dy


#function relu_prime(
#    x::Array{T, N} where {T <: AbstractFloat, N},
#    gradients::Array{T, N} where {T <: AbstractFloat, N}
#)
#    indices = x .> 0
#    dy = zeros(typeof(x), size(gradients))
#    dy[indices] .= gradients[indices]
#    return dy
#end


def sigmoid(x: ndarray):
    y = 1. / (1. + np.exp(-(x - x.max())))
    return y


#function sigmoid(x::Array{T, N} where {T <: AbstractFloat, N})
#    y = 1 ./ (1 + exp(-(x .- max(x...))))
#    return y
#end


def sigmoid_prime(x: ndarray, gradients: ndarray):
    exp_norm_x = np.exp(-(x - x.max()))
    dydx = exp_norm_x / ((1. + exp_norm_x) ** 2.)
    dx = np.matmul(gradients, dydx)
    return dx


#function sigmoid_prime(
#    x::Array{T, N} where {T <: AbstractFloat, N},
#    gradients::Array{T, N} where {T <: AbstractFloat, N}
#)
#    norm_x = x .- max(x...)
#    dydx = exp(-norm_x) / ((1 .+ exp(-norm_x)) .^ 2)
#    dx = gradients * dydx
#    return dx
#end


def softmax(x: ndarray):
    exponents = np.exp(x - x.max())  # normalized exponents.
    y = exponents / exponents.sum()
    return y


#function softmax(x::Vector{T} where T <: AbstractFloat)
#    exponents = exp(x .- max(x...))  # normalized exponents.
#    y = exponents ./ sum(exponents)
#    return y
#end


def softmax_prime(x: ndarray, gradients: ndarray):
    y = softmax(x)
    dydx = np.diag(y) - np.matmul(y, y.T)
    dx = np.matmul(gradients, dydx)
    return dx


#function softmax_prime(
#    x::Vector{T} where T <: AbstractFloat,
#    gradients::Vector{T} where T <: AbstractFloat
#)
#    y = softmax(x)
#    dydx = diagm(0 => y) - (s * s')
#    dx = gradients * dydx
#    return dx
#end


def squared_distance(x: ndarray, y: ndarray):
    assert x.ndim == 1, 'Input array must be of rank 1 (vector).'
    assert x.shape == y.shape, 'Input vectors dimensions does not match.'

    delta = (x - y) ** 2.
    return delta


#function square_distance(
#    x::Vector{T} where T <: AbstractFloat,
#    y::Vector{T} where T <: AbstractFloat
#)
#    if size(x) != size(y)
#        throw(DimensionMismatch("Input vectors dimensions does not match."))
#    end

#    e = (x - y) .^ 2
#    return e
#end


def squared_distance_prime(x: ndarray, y: ndarray, gradients: ndarray) -> Tuple[ndarray, ndarray]:
    assert x.ndim == 1, 'Input array must be of rank 1 (vector).'
    assert x.shape == y.shape, 'Input vectors dimensions does not match.'

    dydx = (2. * (x - y)).T
    dx = np.matmul(gradients, dydx)
    return dx, dx


#function square_distance_prime(
#    x::Vector{T} where T <: AbstractFloat,
#    y::Vector{T} where T <: AbstractFloat,
#    gradients::Vector{T} where T <: AbstractFloat
#)
#    if size(x) != size(y)
#        throw(DimensionMismatch("Input vectors dimensions does not match."))
#    end

#    dydx = 2 .* (x - y)
#    dx = gradients * dydx'
#    return dx
#end


def cross_entropy(x: ndarray, y: ndarray):
    assert x.ndim == 1, 'Input array must be of rank 1 (vector).'
    assert x.shape == y.shape, 'Input vectors dimensions does not match.'

    max_x = x.max()
    norm_logsumexp = max_x + np.log(np.exp(x - max_x).sum())
    e = -np.dot(y, x - norm_logsumexp)
    return e


#function cross_entropy(
#    x::Vector{T} where T <: AbstractFloat,
#    y::Vector{T} where T <: AbstractFloat
#)
#    if size(x) != size(y)
#        throw(DimensionMismatch("Input vectors dimensions does not match."))
#    end

#    max_x = max(x...)
#    norm_logsumexp = max_x + log(sum(exp(x .- max_x)))
#    e = -dot(y, x .- norm_logsumexp)
#    return e
#end


def cross_entropy_prime(x: ndarray, y: ndarray, gradients: ndarray):
    assert x.ndim == 1, 'Input array must be of rank 1 (vector).'
    assert x.shape == y.shape, 'Input vectors dimensions does not match.'

    dydx = x - y
    dx = np.matmul(gradients, dydx)
    return dx.T if dx.shape[0] > 1 else dx


#function cross_entropy_prime(
#    x::Vector{T} where T <: AbstractFloat,
#    y::Vector{T} where T <: AbstractFloat,
#    gradients::Vector{T} where T <: AbstractFloat
#)
#    if size(x) != size(y)
#        throw(DimensionMismatch("Input vectors dimensions does not match."))
#    end

#    dxdy = x - y
#    dx = gradients * dxdy
#    if size(dx, 1) > 1; return dx'; end
#    return dx
#end

