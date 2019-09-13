import abc
from typing import Optional, Sequence, Union, Tuple, Dict, Type, Iterable, List

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core import Variable, SingularVariable
from edunet.core.math import matmul, squared_distance, squared_distance_prime
from edunet.core.initializers import Initializer, HeNormal, Zeros, Ones


class Operation(abc.ABC):
    def __init__(
            self,
            inputs: list,
            var_list: List[Variable],
            shape: Sequence[int],
            dtype: Union[type, np.dtype],
            batch_size: int
    ):
        self._inputs: List[Operation] = inputs

        self.outputs: Variable = Variable(None, shape, dtype, batch_size)
        self.var_list = var_list
        self.grads_dict: Optional[Dict[Variable, Variable]] = dict()

    @property
    def inputs(self):
        return self._inputs

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def backward(self, gradients: Variable = None):
        pass


class Input(Operation):
    def __init__(self, shape, dtype, batch_size: int = 1):
        super(Input, self).__init__(list(), list(), shape, dtype, batch_size)

    def feed(self, values: Sequence[ndarray]):
        for i, value in enumerate(values):
            self.outputs.set_value(i, value)

    def forward(self):
        return self.outputs.as_tuple()

    def backward(self, gradients: Variable = None):
        pass


class DefaultInput(Input):
    def __init__(self, values: Sequence[ndarray]):
        n_values = len(values)
        if n_values == 0:
            raise ValueError('Values container can not be empty.')

        value = values[0]
        super().__init__(value.shape, value.dtype, n_values)
        self.feed(values)


def feed_pairs(feed_dict: Dict[Input, ndarray]):
    for layer, value in feed_dict.items():
        layer.feed(value)


class Dense(Operation):
    def __init__(
            self,
            input_layer: Operation,
            units: int,
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = Zeros,
            trainable: bool = True,
            random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None,
    ):
        input_shape = input_layer.outputs.shape
        input_type = input_layer.outputs.dtype
        batch_size = input_layer.outputs.nitems

        if len(input_shape) != 1 and not (len(input_shape) == 2 and (input_shape[0] == 1 or input_shape[1] == 1)):
            raise ValueError('Input layer must be 1-D.')

        weights_shape = (units, input_shape[0])
        bias_shape = (units, 1)

        w = weights_initializer(weights_shape, input_type).initialize(random_state)
        weights = SingularVariable.from_variable(w)
        b = bias_initializer(bias_shape, input_type).initialize(random_state)
        bias = SingularVariable.from_variable(b)

        output_shape = (units, 1)

        inputs = [input_layer]
        var_list = [weights, bias] if trainable else []

        super(Dense, self).__init__(inputs, var_list, output_shape, input_type, batch_size)
        self.__input_layer = input_layer
        self.__weights = weights
        self.__bias = bias
        self.__trainable = trainable

    def forward(self):
        inputs = self.__input_layer.outputs
        for i in range(inputs.nitems):
            outputs = matmul(self.__weights.get_values(), inputs.get_values(i)) + self.__bias.get_values()
            self.outputs.set_value(i, outputs)
        return self.outputs.as_tuple()

    def backward(self, gradients: Variable = None):
        grads = np.array(1.0, self.outputs.dtype) if gradients is None else gradients.outputs.get_values()

        dx = list()
        dw = list()
        db = list()

        for i in range(self.outputs.nitems):
            input_values = self.__input_layer.outputs.get_values(i)

            # Compute layer gradients based on its inputs.
            dydx = self.__weights.get_values()
            dydw = input_values
            dydb = np.ones((1, 1), self.outputs.dtype)

            # Apply chain rule to compute network based layer gradients.

            # TODO: do something with None gradients.
            dx_ = matmul(grads, dydx)
            dw_ = matmul(dydw, grads).T
            db_ = matmul(dydb, grads).T

            dx.append(np.reshape(dx_, self.__input_layer.outputs.shape))
            dw.append(np.reshape(dw_, self.__weights.shape))
            db.append(np.reshape(db_, self.__bias.shape))

        dx = Variable(dx)
        dw = Variable(dw)
        db = Variable(db)

        self.grads_dict[self.__input_layer.outputs] = dx
        self.grads_dict[self.__weights] = dw
        self.grads_dict[self.__bias] = db

        return str((
            self.grads_dict[self.__input_layer.outputs].as_tuple(),
            self.grads_dict[self.__weights].as_tuple(),
            self.grads_dict[self.__bias].as_tuple()
        ))


class ReduceSum(Operation):
    def __init__(self, input_layer: Operation, reduce_batch: bool = False, axis: int = None, keepdims: bool = False):
        input_shape = input_layer.outputs.shape

        if reduce_batch:
            output_batch_size = 1
            output_shape = input_layer.outputs.shape
        else:
            output_batch_size = input_layer.outputs.nitems
            if axis is None:
                output_shape = (1, 1)
            else:
                output_shape = list(input_shape)
                if keepdims or len(output_shape) == 2:
                    output_shape[axis] = 1
                elif len(output_shape) > 2:
                    output_shape.pop(axis)
                else:
                    raise AssertionError('Bad input variable shape.')
                output_shape = tuple(output_shape)

        inputs = [input_layer]
        var_list = []

        super(ReduceSum, self).__init__(inputs, var_list, output_shape, input_layer.outputs.dtype, output_batch_size)
        self.__input_layer = input_layer
        self.__reduce_batch = reduce_batch
        self.__axis = axis
        self.__keepdims = keepdims

    def forward(self):
        outputs = self.__forward(self.__input_layer.outputs)
        self.outputs.assign(outputs)
        return self.outputs.as_tuple()

    def backward(self, gradients: Variable = None):
        grads = Variable(np.ones([self.outputs.nitems], self.outputs.dtype)) if gradients is None else gradients.outputs

        inputs = self.__input_layer.outputs
        input_values = np.ones([inputs.nitems] + list(inputs.shape), inputs.dtype)
        input_values = Variable(input_values)
        unchained_grads = self.__forward(input_values)

        sum_grads = list()
        for i in range(inputs.nitems):
            outputs = matmul(unchained_grads.get_values(i), grads.get_values(i))
            sum_grads.append(outputs)

        sum_grads = Variable(sum_grads)
        self.grads_dict[self.__input_layer.outputs] = sum_grads

        return str(self.grads_dict[self.__input_layer.outputs])

    def __forward(self, variable: Variable) -> Variable:
        inputs = variable.as_array()
        axis = 0 if self.__reduce_batch else (self.__axis + 1)
        outputs = np.sum(inputs, axis, inputs.dtype, keepdims=self.__keepdims)
        output_variable = Variable([outputs] if self.__reduce_batch and not self.__keepdims else outputs)
        return output_variable

    # def __forward(self, variable: Variable):
    #     outputs = list()
    #     for i in range(variable.nitems):
    #         input_values = variable.get_values(i)
    #         if self.__reduce_batch:
    #             output_values = input_values
    #         else:
    #             output_values = np.sum(input_values, self.__axis, keepdims=self.__keepdims)
    #             output_values = np.reshape(output_values, self.outputs.shape)
    #
    #         outputs.append(output_values)
    #
    #     if self.__reduce_batch:
    #         outputs = np.sum(outputs, 0, keepdims=True)
    #
    #     outputs = Variable(outputs)
    #     return outputs


class SquaredDistance(Operation):
    def __init__(self, x: Operation, y: Operation):
        assert x.outputs.shape == y.outputs.shape, 'Logits and labels shapes must match.'
        assert x.outputs.dtype == y.outputs.dtype, 'Logits and labels dtypes must match.'
        assert x.outputs.nitems == y.outputs.nitems, 'Logits and labels batch_size must match.'

        inputs = [x, y]
        var_list = []

        super(SquaredDistance, self).__init__(inputs, var_list, (1, 1), x.outputs.dtype, x.outputs.nitems)
        self.__x = x
        self.__y = y

    def forward(self):
        for i in range(self.outputs.nitems):
            flat_x = np.reshape(self.__x.outputs.get_values(i), [np.prod(self.__x.outputs.shape)])
            flat_y = np.reshape(self.__y.outputs.get_values(i), [np.prod(self.__y.outputs.shape)])
            outputs = squared_distance(flat_x, flat_y)
            outputs = np.reshape(outputs, self.outputs.shape)
            self.outputs.set_value(i, outputs)

        return str(self.outputs.as_tuple())

    def backward(self, gradients: Variable = None):
        grads = np.ones((1, 1), self.outputs.shape) if gradients is None else gradients.values

        x_grads = list()
        y_grads = list()

        for i in range(self.outputs.nitems):
            flat_x = np.reshape(self.__x.outputs.get_values(i), [np.prod(self.__x.outputs.shape)])
            flat_y = np.reshape(self.__y.outputs.get_values(i), [np.prod(self.__y.outputs.shape)])
            flat_x_grads, flat_y_grads = squared_distance_prime(flat_x, flat_y, grads)

            x_grads.append(np.reshape(flat_x_grads, self.__x.outputs.shape))
            y_grads.append(np.reshape(flat_y_grads, self.__y.outputs.shape))

        x_grads = Variable(x_grads)
        y_grads = Variable(y_grads)

        self.grads_dict[self.__x.outputs] = x_grads
        self.grads_dict[self.__y.outputs] = y_grads

        return str((
            self.grads_dict[self.__x.outputs].as_tuple(),
            self.grads_dict[self.__y.outputs].as_tuple()
        ))


def test():
    def create_algebra_dataset(n: int, seed: int = None):
        rand = np.random.RandomState(seed)
        x1 = rand.uniform(size=(n, 1))
        x2 = rand.uniform(size=(n, 1))
        y = x1 / 3. + x2 / 7. - 2.
        return x1, x2, y

    batch_size = 4

    X1, X2, Y = create_algebra_dataset(batch_size)

    x = [np.array([X1[i], X2[i]], np.float32) for i in range(batch_size)]
    y = [np.float32(np.reshape(Y[i], (1, 1))) for i in range(batch_size)]

    x = np.stack(x)
    y = np.stack(y)

    input_data = Input([2, 1], np.float32, batch_size)
    input_labels = Input([1, 1], np.float32, batch_size)
    dense = Dense(input_data, 1)
    loss = SquaredDistance(dense, input_labels)
    reduce_sum = ReduceSum(loss, reduce_batch=True)

    feed_pairs({
        input_data: x,
        input_labels: y,
    })

    input_data.forward()
    dense.forward()
    input_labels.forward()
    loss.forward()
    reduce_sum.forward()

    reduce_sum.backward(None)



    # loss.backward(None)
    # input_labels.backward(loss.grads_dict[input_labels.outputs])
    # dense.backward(loss.grads_dict[dense.outputs])
    # input_data.backward(dense.grads_dict[input_data.outputs])

    pass


# test()




# class Operation(abc.ABC):
#     def __init__(self, inputs: list, var_list: List[Variable], shape: Sequence[int], dtype: Union[type, np.dtype]):
#         self._inputs: List[Operation] = inputs
#         self._shape: Tuple[int, ...] = tuple(shape)
#         self._dtype: np.dtype = np.dtype(dtype)
#
#         self.outputs: Variable = Variable()
#         self.var_list = var_list
#         self.grads_dict: Optional[Dict[Variable, Variable]] = dict()
#
#     @property
#     def inputs(self):
#         return self._inputs
#
#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self._shape
#
#     @property
#     def dtype(self) -> np.dtype:
#         return self._dtype
#
#     def clear_cache(self):
#         self.outputs.values = None
#         self.grads_dict.clear()
#
#     @abc.abstractmethod
#     def forward(self):
#         pass
#
#     @abc.abstractmethod
#     def backward(self, gradients: Variable = None):
#         pass
#
#
# class Constant(Operation):
#     def __init__(self, values: ndarray):
#         super(Constant, self).__init__(list(), list(), values.shape, values.dtype)
#         self.outputs.values = values
#
#     def forward(self):
#         return self.outputs.values
#
#     def backward(self, gradients: Variable = None):
#         pass
#
#
# class Input(Operation):
#     def __init__(self, shape, dtype):
#         super(Input, self).__init__(list(), list(), shape, dtype)
#
#     def feed(self, values: ndarray):
#         if self._shape != values.shape:
#             raise ValueError('Given value dimensions must match that of the layer.')
#         if self._dtype != values.dtype:
#             raise TypeError('Value `dtype` must match that of the defined layer.')
#         self.outputs.values = values
#
#     def forward(self):
#         return self.outputs.values
#
#     def backward(self, gradients: Variable = None):
#         pass
#
#
# class DefaultInput(Input):
#     def __init__(self, values: ndarray):
#         super().__init__(values.shape, values.dtype)
#         self.feed(values)
#
#
# class Dense(Operation):
#     def __init__(
#             self,
#             input_layer: Operation,
#             units: int,
#             weights_initializer: Type[Initializer] = HeNormal,
#             bias_initializer: Type[Initializer] = Zeros,
#             trainable: bool = True,
#             random_state: Union[None, int, ndarray, Iterable, float, RandomState] = None,
#     ):
#         input_type = input_layer.dtype
#         input_shape = input_layer.shape
#
#         if len(input_shape) != 1 and not (len(input_shape) == 2 and (input_shape[0] == 1 or input_shape[1] == 1)):
#             raise ValueError('Input layer must be 1-D.')
#
#         weights_shape = (units, input_shape[0])
#         bias_shape = (units, 1)
#
#         weights = weights_initializer(weights_shape, input_type).initialize(random_state)
#         bias = bias_initializer(bias_shape, input_type).initialize(random_state)
#
#         output_shape = (units, 1)
#
#         inputs = [input_layer]
#         var_list = [weights, bias] if trainable else []
#
#         super(Dense, self).__init__(inputs, var_list, output_shape, input_type)
#         self.__input_layer = input_layer
#         self.__weights = weights
#         self.__bias = bias
#         self.__trainable = trainable
#
#     def forward(self):
#         inputs = self.__input_layer.outputs
#         outputs = np.matmul(self.__weights.values, inputs.values) + self.__bias.values
#         self.outputs.values = outputs
#         return outputs
#
#     def backward(self, gradients: Variable = None):
#         grads = np.ones((1, 1), self._dtype) if gradients is None else gradients.values
#
#         # Compute layer gradients based on its inputs.
#         dydx = self.__weights.values
#         dydw = self.__input_layer.outputs.values
#         dydb = np.ones([1, 1], self._dtype)
#
#         # Apply chain rule to compute network based layer gradients.
#         dx = np.matmul(grads, dydx)
#         dw = np.matmul(dydw, grads).T
#         db = np.matmul(dydb, grads).T
#
#         self.grads_dict[self.__input_layer.outputs] = Variable(np.reshape(dx, self.__input_layer.shape))
#         self.grads_dict[self.__weights] = Variable(np.reshape(dw, self.__weights.shape))
#         self.grads_dict[self.__bias] = Variable(np.reshape(db, self.__bias.shape))
#
#         return (
#             self.grads_dict[self.__input_layer.outputs],
#             self.grads_dict[self.__weights],
#             self.grads_dict[self.__bias]
#         )
#
#
# class SquaredDistance(Operation):
#     def __init__(self, x: Operation, y: Operation):
#         assert x.shape == y.shape, 'Logits and labels shapes must match.'
#         assert x.dtype == y.dtype, 'Logits and labels dtypes must match'
#
#         inputs = [x, y]
#         var_list = []
#
#         super(SquaredDistance, self).__init__(inputs, var_list, (1, 1), x.dtype)
#         self.__x = x
#         self.__y = y
#
#     def forward(self):
#         flat_x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.shape)])
#         flat_y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.shape)])
#         outputs = squared_distance(flat_x, flat_y)
#         outputs = np.reshape(outputs, self._shape)
#         self.outputs.values = outputs
#         return outputs
#
#     def backward(self, gradients: Variable = None):
#         grads = np.ones((1, 1), self._dtype) if gradients is None else gradients.values
#
#         flat_x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.shape)])
#         flat_y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.shape)])
#         flat_x_grads, flat_y_grads = squared_distance_prime(flat_x, flat_y, grads)
#
#         self.grads_dict[self.__x.outputs] = Variable(np.reshape(flat_x_grads, self.__x.shape))
#         self.grads_dict[self.__y.outputs] = Variable(np.reshape(flat_y_grads, self.__y.shape))
#
#         return self.grads_dict[self.__x.outputs].values, self.grads_dict[self.__y.outputs].values
#
#
# # class Reshape(Operation):
# #     def __init__(self, input_layer: Operation, shape: Sequence[int]):
# #         super(Reshape, self).__init__([input_layer], shape, input_layer.dtype)
# #         self.__input_layer = input_layer
# #
# #     def forward(self):
# #         inputs = self.__input_layer.outputs
# #         outputs = np.reshape(inputs.values, self.shape)
# #         self.outputs = Variable(outputs)
# #         return outputs
# #
# #     def backward(self):
# #         # backprop_outputs = np.reshape(self.cache.forward_results, self.__input_layer.shape)
# #         # self._cache.backward_results = backprop_outputs
# #         # return backprop_outputs
# #         pass
# #
# #
# # class Softmax(Operation):
# #     def __init__(self, input_layer: Operation):
# #         super(Softmax, self).__init__([input_layer], input_layer.shape, input_layer.dtype)
# #         self.__input_layer = input_layer
# #
# #     def forward(self):
# #         inputs = self.__input_layer.outputs
# #         flat_inputs = np.reshape(inputs.values, [np.prod(self._shape)])
# #         flat_outputs = edumath.softmax(flat_inputs)
# #         outputs = np.reshape(flat_outputs, inputs.shape)
# #         self.outputs = Variable(outputs)
# #         return outputs
# #
# #     def backward(self):
# #         pass
# #
# #
# # class CrossEntropy(Operation):
# #     def __init__(self, logits: Operation, labels: Operation):
# #         assert logits.shape == labels.shape, 'Logits and labels shapes must match.'
# #         assert logits.dtype == labels.dtype, 'Logits and labels dtypes must match'
# #
# #         super().__init__([logits, labels], (1, 1), logits.dtype)
# #         self.__logits = logits
# #         self.__labels = labels
# #
# #     def forward(self):
# #         flat_logits = np.reshape(self.__logits.outputs.values, [np.prod(self.__logits.shape)])
# #         flat_labels = np.reshape(self.__labels.outputs.values, [np.prod(self.__labels.shape)])
# #         outputs = edumath.cross_entropy(flat_logits, flat_labels)
# #         outputs = np.reshape(outputs, self._shape)
# #         self.outputs = Variable(outputs)
# #         return outputs
# #
# #     def backward(self):
# #         pass
