from abc import ABC
from typing import Optional, Sequence, Union, Tuple, Dict, Type, Iterable, Any, List
import abc
import random
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

import edunet.math as edumath
from edunet.variable import Variable
from edunet.initializers import Initializer
from edunet.initializers import HeNormal


class Operation(abc.ABC):
    def __init__(self, inputs: list, var_list: List[Variable], shape: Sequence[int], dtype: Union[type, np.dtype]):
        self._inputs: List[Operation] = inputs
        self._shape: Tuple[int, ...] = tuple(shape)
        self._dtype: np.dtype = np.dtype(dtype)

        self.outputs: Variable = Variable()
        self.var_list = var_list
        self.grads_dict: Optional[Dict[Variable, Variable]] = dict()

    @property
    def inputs(self):
        return self._inputs

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def backward(self, gradients: Variable = None):
        pass


class Input(Operation):
    def __init__(self, shape, dtype):
        self.__inputs: Variable = Variable()
        super(Input, self).__init__(list(), [self.__inputs], shape, dtype)

    def feed(self, values: ndarray):
        if self._shape != values.shape:
            raise ValueError('Given value dimensions must match that of the layer.')
        if self._dtype != values.dtype:
            raise TypeError('Value `dtype` must match that of the defined layer.')
        self.__inputs.values = values

    def forward(self):
        self.outputs.assign(self.__inputs)
        return self.__inputs

    def backward(self, gradients=None):
        pass


def feed_pairs(feed_dict: Dict[Input, ndarray]):
    for layer, value in feed_dict.items():
        layer.feed(value)


class Dense(Operation):
    def __init__(
            self,
            input_layer: Operation,
            units: int,
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = HeNormal,
            trainable: bool = True,
            random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None,
    ):
        input_type = input_layer.dtype
        input_shape = input_layer.shape

        if len(input_shape) != 1 and not (len(input_shape) == 2 and (input_shape[0] == 1 or input_shape[1] == 1)):
            raise ValueError('Input layer must be 1-D.')

        weights_shape = (units, input_shape[0])
        bias_shape = (units, 1)

        weights = weights_initializer(weights_shape, input_type).initialize(random_state)
        bias = bias_initializer(bias_shape, input_type).initialize(random_state)

        output_shape = (units, 1)

        inputs = [input_layer]
        var_list = [weights, bias]

        super(Dense, self).__init__(inputs, var_list, output_shape, input_type)
        self.__input_layer = input_layer
        self.__weights = weights
        self.__bias = bias
        self.__trainable = trainable

    def forward(self):
        inputs = self.__input_layer.outputs
        outputs = np.matmul(self.__weights.values, inputs.values) + self.__bias.values
        self.outputs.values = outputs
        return outputs

    def backward(self, gradients=None):
        grads = np.ones((1, 1), self._dtype) if gradients is None else gradients.values

        # Compute layer gradients based on its inputs.
        dydx = self.__weights.values
        dydw = self.__input_layer.outputs.values
        dydb = np.ones([1, 1], self._dtype)

        # Apply chain rule to compute network based layer gradients.
        dx = np.matmul(grads, dydx)
        dw = np.matmul(dydw, grads).T
        db = np.matmul(dydb, grads).T

        self.grads_dict[self.__input_layer.outputs] = Variable(np.reshape(dx, self.__input_layer.shape))
        self.grads_dict[self.__weights] = Variable(np.reshape(dw, self.__weights.shape))
        self.grads_dict[self.__bias] = Variable(np.reshape(db, self.__bias.shape))

        return (
            self.grads_dict[self.__input_layer.outputs],
            self.grads_dict[self.__weights],
            self.grads_dict[self.__bias]
        )


# class Reshape(Operation):
#     def __init__(self, input_layer: Operation, shape: Sequence[int]):
#         super(Reshape, self).__init__([input_layer], shape, input_layer.dtype)
#         self.__input_layer = input_layer
#
#     def forward(self):
#         inputs = self.__input_layer.outputs
#         outputs = np.reshape(inputs.values, self.shape)
#         self.outputs = Variable(outputs)
#         return outputs
#
#     def backward(self):
#         # backprop_outputs = np.reshape(self.cache.forward_results, self.__input_layer.shape)
#         # self._cache.backward_results = backprop_outputs
#         # return backprop_outputs
#         pass


# class Softmax(Operation):
#     def __init__(self, input_layer: Operation):
#         super(Softmax, self).__init__([input_layer], input_layer.shape, input_layer.dtype)
#         self.__input_layer = input_layer
#
#     def forward(self):
#         inputs = self.__input_layer.outputs
#         flat_inputs = np.reshape(inputs.values, [np.prod(self._shape)])
#         flat_outputs = edumath.softmax(flat_inputs)
#         outputs = np.reshape(flat_outputs, inputs.shape)
#         self.outputs = Variable(outputs)
#         return outputs
#
#     def backward(self):
#         pass
#
#
# class CrossEntropy(Operation):
#     def __init__(self, logits: Operation, labels: Operation):
#         assert logits.shape == labels.shape, 'Logits and labels shapes must match.'
#         assert logits.dtype == labels.dtype, 'Logits and labels dtypes must match'
#
#         super().__init__([logits, labels], (1, 1), logits.dtype)
#         self.__logits = logits
#         self.__labels = labels
#
#     def forward(self):
#         flat_logits = np.reshape(self.__logits.outputs.values, [np.prod(self.__logits.shape)])
#         flat_labels = np.reshape(self.__labels.outputs.values, [np.prod(self.__labels.shape)])
#         outputs = edumath.cross_entropy(flat_logits, flat_labels)
#         outputs = np.reshape(outputs, self._shape)
#         self.outputs = Variable(outputs)
#         return outputs
#
#     def backward(self):
#         pass


class SquaredDistance(Operation):
    def __init__(self, x: Operation, y: Operation):
        assert x.shape == y.shape, 'Logits and labels shapes must match.'
        assert x.dtype == y.dtype, 'Logits and labels dtypes must match'

        inputs = [x, y]
        var_list = []

        super().__init__(inputs, var_list, (1, 1), x.dtype)
        self.__x = x
        self.__y = y

    def forward(self):
        flat_x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.shape)])
        flat_y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.shape)])
        outputs = edumath.squared_distance(flat_x, flat_y)
        outputs = np.reshape(outputs, self._shape)
        self.outputs.values = outputs
        return outputs

    def backward(self, gradients=None):
        grads = np.ones((1, 1), self._dtype) if gradients is None else gradients.values

        flat_x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.shape)])
        flat_y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.shape)])
        flat_x_grads, flat_y_grads = edumath.squared_distance_prime(flat_x, flat_y, grads)

        self.grads_dict[self.__x.outputs] = Variable(np.reshape(flat_x_grads, self.__x.shape))
        self.grads_dict[self.__y.outputs] = Variable(np.reshape(flat_y_grads, self.__y.shape))

        return self.grads_dict[self.__x.outputs].values, self.grads_dict[self.__y.outputs].values


class FlowGraph(object):
    def __init__(self):
        self.__ops: List[Operation] = list()

    @property
    def ops(self) -> List[Operation]:
        return self.__ops

    def add(self, ops: Union[Operation, Iterable[Operation]]) -> Optional[Operation]:
        if isinstance(ops, Iterable):
            self.__ops.extend(ops)
        else:
            self.__ops.append(ops)
            return ops

    def sort(self):
        sorted_ops = list()
        used_ops = set()

        def __sort(ops: List[Operation]):
            for op in ops:
                if op in used_ops:
                    continue
                if len(op.inputs) > 0:
                    __sort(op.inputs)
                sorted_ops.append(op)
                used_ops.add(op)

        __sort(self.__ops)
        self.__ops = sorted_ops


class Optimizer(abc.ABC):
    def __init__(self):
        pass


class GradientDescent(Optimizer):
    def __init__(self):
        super(GradientDescent, self).__init__()


class FlowControl(object):
    def __init__(self, graph: FlowGraph = None):
        self.__graph = graph
        self.__graph.sort()

    def run_forward(self):
        for op in self.__graph.ops:
            op.forward()

    def run_backward(self):
        for op in self.__graph.ops[::-1]:
            op.backward()

    def run(
            self,
            outputs: List[Operation] = None,
            feed_dict: Dict[Input, ndarray] = None,
            backprop=False
    ) -> Optional[List[Optional[ndarray]]]:
        if feed_dict is not None:
            feed_pairs(feed_dict)

        self.run_forward()

        if backprop:
            self.run_backward()

        if outputs:
            return [op.outputs.values for op in outputs]

    @staticmethod
    def feed(feed_dict: Dict[Operation, ndarray]):
        for op, values in feed_dict.items():
            if not isinstance(op, Input):
                raise TypeError('Feed dictionary key operations must be Input class instances.')
            op.feed(values)


def test():
    def create_algebra_dataset(n: int, seed: int = None):
        rand = np.random.RandomState(seed)
        x1 = rand.uniform(size=(n, 1))
        x2 = rand.uniform(size=(n, 1))
        y = x1 / 3. + x2 / 7. - 2.
        return x1, x2, y

    X1, X2, Y = create_algebra_dataset(4)

    i = 0
    x = np.array([X1[i], X2[i]], np.float32)
    y = np.float32(np.reshape(Y[i], (1, 1)))

    graph = FlowGraph()

    input_data = graph.add(Input((2, 1), np.float32))
    input_labels = graph.add(Input((1, 1), np.float32))

    dense_1 = graph.add(Dense(input_data, 1))

    loss = graph.add(SquaredDistance(dense_1, input_labels))

    control = FlowControl(graph)
    FlowControl.feed({
        input_data: x,
        input_labels: y,
    })
    control.run_forward()
    control.run_backward()

    # input_data.feed(x)
    # input_labels.feed(y)
    #
    # # ---  Feed-forward  -----------------------------------------------------------------------------------------------
    #
    # input_data.forward()
    # input_labels.forward()
    #
    # print(dense_1.forward().shape)
    # print(loss.forward().shape)
    #
    #
    # # ---  Backprop  ---------------------------------------------------------------------------------------------------
    #
    # print('\nloss back:')
    # for variable in loss.backward():
    #     print(variable.shape)
    #
    # print('\ndense_1 back:')
    # for variable in dense_1.backward():
    #     print(variable.shape)

    pass


test()
