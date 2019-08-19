from typing import Tuple, Optional

import numpy as np
from numpy import ndarray


class Variable(object):
    def __init__(self, values: ndarray = None):
        self.__values = values

    @property
    def values(self) -> Optional[ndarray]:
        return self.__values

    @values.setter
    def values(self, a: ndarray):
        self.__values = a

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__values.shape

    @property
    def dtype(self) -> np.dtype:
        return self.__values.dtype

    def is_empty(self) -> bool:
        return self.__values is None

    def assign(self, other):
        if not isinstance(other, Variable):
            raise TypeError('Argument `other` must be an instance of `Variable` class.')
        self.__values = other.values
