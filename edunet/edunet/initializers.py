import abc
from typing import Union, Iterable

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

import edunet.math as edumath
from edunet.variable import Variable


class Initializer(abc.ABC):
    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    @abc.abstractmethod
    def initialize(self, random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None) -> Variable:
        pass


class HeNormal(Initializer):
    def initialize(self, random_state=None):
        return Variable(edumath.he_normal(
            self._shape,
            self._shape[0],
            dtype=self._dtype,
            random_state=random_state))


class HeUniform(Initializer):
    def initialize(self, random_state=None):
        return Variable(edumath.he_uniform(
            self._shape,
            self._shape[0],
            dtype=self._dtype,
            random_state=random_state))


class RandomNormal(Initializer):
    def initialize(self, random_state=None):
        return Variable(edumath.random_normal(
            self._shape,
            dtype=self._dtype,
            random_state=random_state))


class RandomUniform(Initializer):
    def initialize(self, random_state=None):
        return Variable(edumath.random_uniform(
            self._shape,
            dtype=self._dtype,
            random_state=random_state))


class Ones(Initializer):
    def initialize(self, random_state=None):
        return Variable(np.ones(self._shape, self._dtype))


class Zeros(Initializer):
    def initialize(self, random_state=None):
        return Variable(np.zeros(self._shape, self._dtype))
