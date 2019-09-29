from typing import Sequence, Iterable, Union, Tuple
import math

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core.utilities import isscalar


__all__ = [
    'matmul',
    'random_uniform',
    'random_normal',
    'he_uniform',
    'he_normal',
    'relu',
    'relu_prime',
    'sigmoid',
    'sigmoid_prime',
    'softargmax',
    'softargmax_prime',
    'squared_distance',
    'squared_distance_prime',
    'cross_entropy',
    'cross_entropy_prime',
]


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


def softargmax(x: ndarray, axis: int):
    exponents = np.exp(x - x.max(axis, keepdims=True))
    y = exponents / exponents.sum(axis, keepdims=True)
    return y


def softargmax_prime(x: ndarray, axis: int, gradients: ndarray):
    assert x.shape == gradients.shape, 'Gradients and input arrays shapes must match.'

    y = softargmax(x, axis)
    yy = np.expand_dims(y.swapaxes(axis, -1), -1)
    dydx = yy * np.eye(y.shape[axis], dtype=yy.dtype) - yy * yy.swapaxes(-1, -2)
    dx = np.matmul(dydx, np.expand_dims(gradients.swapaxes(axis, -1), -1)).squeeze(-1).swapaxes(axis, -1)
    return dx


def squared_distance(x: ndarray, y: ndarray):
    assert x.shape == y.shape, 'Input arrays dimensions does not match.'

    e = (x - y) ** 2.
    return e


def squared_distance_prime(x: ndarray, y: ndarray, gradients: ndarray) -> Tuple[ndarray, ndarray]:
    assert x.shape == y.shape, 'Input arrays dimensions does not match.'
    assert x.shape == gradients.shape, 'Input and gradients arrays shapes must match.'

    de = 2. * (x - y)
    dx = de * gradients
    dy = -dx
    return dx, dy


def cross_entropy(x: ndarray, y: ndarray, axis: int):
    assert x.shape == y.shape, 'Input arrays dimensions does not match.'

    e = -np.sum((y * np.log(x)), axis, keepdims=True)
    return e


def cross_entropy_prime(x: ndarray, y: ndarray, axis: int, gradients: ndarray):
    assert y.shape == x.shape, 'Input arrays dimensions does not match.'
    y_shape = list(y.shape)
    y_shape[axis] = 1
    assert tuple(y_shape) == gradients.shape, 'Gradients array shape mismatch.'

    dx = (-y / x) * gradients
    dy = -np.log(x) * gradients
    return dx, dy
