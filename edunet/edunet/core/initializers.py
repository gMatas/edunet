import abc
from typing import Union, Iterable, Sequence

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core_v2 import Variable
from edunet.core_v2.math import random_uniform
from edunet.core_v2.math import random_normal
from edunet.core_v2.math import he_uniform
from edunet.core_v2.math import he_normal


class Initializer(abc.ABC):
    def __init__(self, shape: Sequence[int], dtype: Union[type, np.dtype]):
        self._shape = shape
        self._dtype = dtype

    @abc.abstractmethod
    def initialize(self, random_state: Union[int, ndarray, Iterable, float, RandomState] = None) -> Variable:
        pass


class HeNormal(Initializer):
    def initialize(self, random_state=None):
        return Variable(he_normal(
            self._shape,
            int(np.prod(self._shape[1:])),
            dtype=self._dtype,
            random_state=random_state))


class HeUniform(Initializer):
    def initialize(self, random_state=None):
        return Variable(he_uniform(
            self._shape,
            int(np.prod(self._shape[1:])),
            dtype=self._dtype,
            random_state=random_state))


class RandomNormal(Initializer):
    def initialize(self, random_state=None):
        return Variable(random_normal(
            self._shape,
            dtype=self._dtype,
            random_state=random_state))


class RandomUniform(Initializer):
    def initialize(self, random_state=None):
        return Variable(random_uniform(
            self._shape,
            dtype=self._dtype,
            random_state=random_state))


class Ones(Initializer):
    def initialize(self, **kwargs):
        return Variable(np.ones(self._shape, self._dtype))


class Zeros(Initializer):
    def initialize(self, **kwargs):
        return Variable(np.zeros(self._shape, self._dtype))


class Full(Initializer):
    def __init__(self, shape: Sequence[int], value: Union[int, float, complex], dtype: Union[type, np.dtype]):
        super().__init__(shape, dtype)
        self.__value = value

    def initialize(self, **kwargs) -> Variable:
        return Variable(np.full(self._shape, self.__value, self._dtype))
