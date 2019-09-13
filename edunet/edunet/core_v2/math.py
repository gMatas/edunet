from typing import Sequence, Iterable, Union, Tuple
import math

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core_v2.utilities import isscalar


def matmul(x1: ndarray, x2: ndarray, *args, **kwargs) -> ndarray:
    if isscalar(x1) or isscalar(x2):
        return x1 * x2
    return np.matmul(x1, x2, *args, **kwargs)


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


def relu(x: ndarray):
    y = x.copy()
    y[x < 0] = 0
    return y


def relu_prime(x: ndarray, gradients: ndarray):
    indices = x > 0
    dy = np.zeros(gradients.shape, gradients.dtype)
    dy[indices] = gradients[indices]
    return dy


def sigmoid(x: ndarray):
    y = np.empty(x.shape, x.dtype)
    pos_elements = (x >= 0)
    y[pos_elements] = 1. / (1. + np.exp(-x[pos_elements]))
    neg_elements = ~pos_elements
    z = np.exp(x[neg_elements])
    y[neg_elements] = z / (1. + z)
    return y


def sigmoid_prime(x: ndarray, gradients: ndarray):
    y = sigmoid(x)
    dx = y * (1 - y) * gradients
    return dx


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

    e = (x - y) ** 2.
    return e


def squared_distance_prime(x: ndarray, y: ndarray, gradients: ndarray) -> Tuple[ndarray, ndarray]:
    assert x.ndim == 1, 'Input array must be of rank 1 (vector).'
    assert x.shape == y.shape, 'Input vectors dimensions does not match.'
    assert gradients.ndim == 1, 'Gradient array must be of rank 1 (vector).'
    assert x.size == gradients.size, \
        'Gradient array must have the same number of elements as each of the input vectors.'

    de = 2. * (x - y)
    dx = de * gradients
    dy = -dx
    return dx, dy


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
    assert gradients.ndim == 1, 'Gradient array must be of rank 1 (vector).'
    assert x.size == gradients.size, \
        'Gradient array must have the same number of elements as each of the input vectors.'

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

