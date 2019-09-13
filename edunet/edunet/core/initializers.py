import abc
from typing import Union, Iterable, Sequence

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core import Variable
from edunet.core.math import random_uniform
from edunet.core.math import random_normal
from edunet.core.math import he_uniform
from edunet.core.math import he_normal


class Initializer(abc.ABC):
    def __init__(self, shape: Sequence[int], dtype: Union[type, np.dtype], batch_size: int = 1):
        self._shape = shape
        self._dtype = dtype
        self._batch_size = batch_size

    @abc.abstractmethod
    def initialize(self, random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None) -> Variable:
        pass


class HeNormal(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        a = he_normal(
            full_shape,
            self._shape[0],
            dtype=self._dtype,
            random_state=random_state)
        return Variable(a)


class HeUniform(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        return Variable(he_uniform(
            full_shape,
            self._shape[0],
            dtype=self._dtype,
            random_state=random_state))


class RandomNormal(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        return Variable(random_normal(
            full_shape,
            dtype=self._dtype,
            random_state=random_state))


class RandomUniform(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        return Variable(random_uniform(
            full_shape,
            dtype=self._dtype,
            random_state=random_state))


class Ones(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        return Variable(np.ones(full_shape, self._dtype))


class Zeros(Initializer):
    def initialize(self, random_state=None):
        full_shape = [self._batch_size] + list(self._shape)
        return Variable(np.zeros(full_shape, self._dtype))
