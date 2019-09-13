from typing import List, Sequence, Union

import numpy as np

from edunet.core import Variable
from edunet.core.operations import Operation


class Optimizer(Operation):
    def __init__(self, inputs: list, shape: Sequence[int], dtype: Union[type, np.dtype]):
        super().__init__(inputs, list(), shape, dtype)

    def forward(self):
        pass

    def backward(self, gradients: Variable = None):
        pass


class GradientDescent(Optimizer):
    def __init__(self):
        super(GradientDescent, self).__init__()
