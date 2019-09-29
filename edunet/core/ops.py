from typing import Tuple, Sequence, Union, Type, Iterable

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core import Variable, Operation
from edunet.core.math import matmul, cross_entropy, cross_entropy_prime
from edunet.core.math import sigmoid_prime, sigmoid
from edunet.core.math import softargmax, softargmax_prime
from edunet.core.math import squared_distance, squared_distance_prime
from edunet.core.math import relu, relu_prime
from edunet.core.ops_utils import collect_ops
from edunet.core.utilities import compute_padding_size_2d, apply_padding_2d, strip_padding_2d
from edunet.core.utilities import compute_window_slide_size_2d, window_slide_2d
from edunet.core.utilities import dilate_map_2d
from edunet.core.initializers import Initializer, HeNormal, Zeros, Full


__all__ = [
    'Gradients',
    'GradientDescentOptimizer',
    'Input',
    'DefaultInput',
    'Convolution2D',
    'AveragePool2D',
    'Dense',
    'SquaredDistance',
    'CrossEntropy',
    'ReduceSum',
    'Reshape',
    'Flatten',
    'Relu',
    'Sigmoid',
    'SoftArgMax',
]


class Gradients(Operation):
    def __init__(self, dy: Operation, dx: Union[Sequence[Operation], None], name: str = None):
        output_shape = (1,) if dx is None else (len(dx),)
        name = self.__class__.__name__ if name is None else name
        super().__init__([dy], list(), output_shape, object, name)
        self.__dx = dx
        self.__dy = dy

        if dx is None:
            self.__sorted_ops = [dy]
        else:
            for dx_op in dx:
                if dx_op is dy:
                    raise ValueError(
                        'To differentiate operation `dy` in respect to itself set `dx` argument to `None`.')

            self.__sorted_ops = collect_ops(dy, set(dx))

    def run(self):
        self.__gradients_backprop()
        self.output.set_values(
            np.array([self.__dy.grads_dict] if self.__dx is None else [op.grads_dict for op in self.__dx]))

    def compute_gradients(self, gradients: Variable = None):
        pass

    def __gradients_backprop(self):
        prev_op = self.__sorted_ops[-1]
        prev_op.compute_gradients(None)

        for op in self.__sorted_ops[-2::-1]:
            # Check if previous op gradients for current op are set.
            if op.output not in prev_op.grads_dict:
                break

            op.compute_gradients(prev_op.grads_dict[op.output])
            prev_op = op


class GradientDescentOptimizer(object):
    class _Optimizer(Operation):
        def __init__(self, input_op: Operation, learning_rate: Union[int, float], name: str):
            super(GradientDescentOptimizer._Optimizer, self).__init__(
                [input_op], list(), input_op.output.shape, input_op.output.dtype, name)
            self.__input_layer = input_op
            self.__learning_rate = learning_rate

            self._ops = [op for op in collect_ops(input_op) if len(op.var_list) > 0]
            self._gradients_op = Gradients(input_op, self._ops)

        def run(self):
            self._gradients_op.run()
            for op, grads_dict in zip(self._ops, self._gradients_op.output.values):
                for op_var in op.var_list:
                    if op_var not in grads_dict:
                        continue

                    grads_var: Variable = grads_dict[op_var]
                    updated_values = np.mean(self._optimization_operator(
                        op_var.values, self.__learning_rate * grads_var.values), 0)
                    op_var.set_values(updated_values)

        def compute_gradients(self, gradients: Variable = None):
            self._gradients_op.compute_gradients()

        def _optimization_operator(self, a, b) -> ndarray:
            raise NotImplementedError('Optimization operator function must be override by inheritor classes.')

    class Minimizer(_Optimizer):
        def __init__(self, input_op: Operation, learning_rate: Union[int, float], name: str = None):
            name = self.__class__.__name__ if name is None else name
            super(GradientDescentOptimizer.Minimizer, self).__init__(input_op, learning_rate, name)

        def _optimization_operator(self, a, b) -> ndarray:
            return a - b

    class Maximizer(_Optimizer):
        def __init__(self, input_op: Operation, learning_rate: Union[int, float], name: str = None):
            name = self.__class__.__name__ if name is None else name
            super(GradientDescentOptimizer.Maximizer, self).__init__(input_op, learning_rate, name)

        def _optimization_operator(self, a, b) -> ndarray:
            return a + b

    def __init__(self, learning_rate: float = 0.01):
        self.__learning_rate = learning_rate

    def minimize(self, input_op: Operation) -> Minimizer:
        return GradientDescentOptimizer.Minimizer(input_op, self.__learning_rate)

    def maximize(self, input_op: Operation) -> Maximizer:
        return GradientDescentOptimizer.Maximizer(input_op, self.__learning_rate)


class Input(Operation):
    def __init__(self, shape, dtype, name: str = None):
        name = self.__class__.__name__ if name is None else name
        super(Input, self).__init__(list(), list(), shape, dtype, name)

    def feed(self, values: ndarray):
        self.output.set_values(values)

    def run(self):
        pass

    def compute_gradients(self, gradients: Variable = None):
        pass


class DefaultInput(Input):
    def __init__(self, values: ndarray, name: str = None):
        name = self.__class__.__name__ if name is None else name
        super().__init__(values.shape, values.dtype, name)
        self.feed(values)


class Convolution2D(Operation):
    def __init__(
            self,
            input_op: Operation,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            strides: Union[int, Tuple[int, int]] = 1,
            mode: str = 'valid',
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = Zeros,
            trainable: bool = True,
            random_state: Union[int, ndarray, Iterable, float, RandomState] = None,
            name: str = None
    ):
        inputs = input_op.output

        input_shape = inputs.shape  # (BATCH, HEIGHT, WIDTH, DEPTH)
        input_type = inputs.dtype
        batch_size = input_shape[0]

        # Initialize kernel and bias tensors.
        kernel_size_y, kernel_size_x = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        kernel_shape = (filters, kernel_size_y, kernel_size_x, input_shape[3])
        bias_shape = (filters, 1, 1, 1)

        weights = weights_initializer(kernel_shape, input_type).initialize(random_state=random_state)
        bias = bias_initializer(bias_shape, input_type).initialize(random_state=random_state)

        # Set stride parameters.
        stride_y, stride_x = (strides, strides) if isinstance(strides, int) else strides

        mode = mode.lower()
        output_size_y, output_size_x = compute_window_slide_size_2d(
            input_shape[1:3], (kernel_size_y, kernel_size_x), (stride_y, stride_x), mode)
        output_shape = (batch_size, output_size_y, output_size_x, filters)

        inputs = [input_op]
        var_list = [weights, bias]

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, output_shape, input_type, name)
        self.__input_op = input_op
        self.__weights = weights
        self.__bias = bias
        self.__strides: Tuple[int, int] = (stride_y, stride_x)
        self.__mode = mode
        self.__trainable = trainable

    def run(self):
        ksize = self.__weights.shape[1:3]

        padded_input_values = np.moveaxis(self.__input_op.output.values, 0, -1)
        padded_input_values = apply_padding_2d(padded_input_values, self.__mode, ksize, self.__strides)
        strided_input_values = window_slide_2d(padded_input_values, ksize, self.__strides)
        flipped_weights = np.flip(self.__weights.values, (1, 2))
        output_values = np.sum((strided_input_values[None, ...] * flipped_weights[:, None, None, ..., None]), (3, 4, 5))
        output_values = output_values + self.__bias.values
        output_values = output_values.swapaxes(0, -1)

        self.output.set_values(output_values)

    def compute_gradients(self, gradients: Variable = None):
        inputs = self.__input_op.output
        batch_size = inputs.shape[0]
        weights = self.__weights.values
        nfilters = weights.shape[0]

        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values

        # Compute operation gradients in respect to input values.
        grads_h, grads_w = grads.shape[1:3]
        _, dilated_grads_indices = dilate_map_2d(grads_w, grads_h, *self.__strides, grads.dtype)
        dilated_grads = np.zeros((batch_size, dilated_grads_indices.size, nfilters), grads.dtype)
        dilated_grads[:, dilated_grads_indices.flat, :] = grads.reshape([batch_size, (grads_h * grads_w), nfilters])
        dilated_grads = dilated_grads.reshape((batch_size, *dilated_grads_indices.shape, nfilters))

        ksize = weights.shape[1:3]
        _dilated_grads_out_size = compute_window_slide_size_2d(dilated_grads.shape[1:3], ksize, 1, 'full')
        _y_input_pad, _x_input_pad = compute_padding_size_2d(inputs.shape[1:3], ksize, self.__strides, self.__mode)
        _padded_input_size = (
            inputs.shape[1] + int(2 * _y_input_pad),
            inputs.shape[2] + int(2 * _x_input_pad))
        if _dilated_grads_out_size != _padded_input_size:
            dilated_grads_h, dilated_grads_w = dilated_grads.shape[1:3]
            padded_dilated_grads = np.zeros(
                [batch_size, dilated_grads_h + 1, dilated_grads_w + 1, nfilters], dilated_grads.dtype)
            padded_dilated_grads[:, :dilated_grads_h, :dilated_grads_w, :] = dilated_grads
            dilated_grads = padded_dilated_grads

        padded_dilated_grads = np.moveaxis(dilated_grads, 0, -1)
        padded_dilated_grads = apply_padding_2d(padded_dilated_grads, 'full', ksize, 1)
        strided_dilated_grads = window_slide_2d(padded_dilated_grads, ksize, 1)
        dx_padded = np.sum(
            (np.moveaxis(strided_dilated_grads, 4, 0)[..., None] * weights[:, None, None, ..., None, :]),
            (0, 3, 4))
        dx = np.moveaxis(strip_padding_2d(dx_padded, (_y_input_pad, _x_input_pad), self.__mode), 2, 0)

        # Compute operation gradients in respect to weights values.
        padded_input_values = np.moveaxis(inputs.values, 0, -1)
        padded_input_values = apply_padding_2d(padded_input_values, self.__mode, padding=(_y_input_pad, _x_input_pad))
        strided_input_values = window_slide_2d(padded_input_values, ksize, self.__strides)
        dw = np.flip(np.sum(
            (np.moveaxis(strided_input_values, -1, 0)[:, None, ...] * np.moveaxis(grads, -1, 1)[..., None, None, None]),
            (2, 3)), (2, 3))

        # Compute operation gradients in respect to bias values.
        # TODO: Investigate the reason why bias gradients calculated by the fallowing formula do not
        #  match with numerically differentiated gradients while using "same" padding mode. Though,
        #  their directions seems to match.
        db = np.sum(grads, (1, 2))[..., None, None, None]

        # Operation gradients mapping.
        self.grads_dict[inputs] = Variable(dx)
        self.grads_dict[self.__weights] = Variable(dw)
        self.grads_dict[self.__bias] = Variable(db)


class AveragePool2D(Operation):
    def __init__(
            self,
            input_op: Operation,
            size: Union[int, Tuple[int, int]] = 2,
            strides: Union[int, Tuple[int, int], None] = None,
            mode: str = 'valid',
            name: str = None
    ):
        if mode not in {'valid', 'same'}:
            raise ValueError(
                'Sliding window mode for AveragePool2D operation can '
                'only be one of the fallowing: "valid" or "same".')

        size = (size, size) if isinstance(size, int) else tuple(size)
        strides = size if strides is None else (strides, strides) if isinstance(strides, int) else tuple(strides)

        inputs_depth = input_op.output.shape[3]
        weight_value = 1. / np.prod(size)
        kernel_shape = (inputs_depth, *size, inputs_depth)
        weights = Full(kernel_shape, weight_value, input_op.output.dtype).initialize()

        input_shape = input_op.output.shape  # (BATCH, HEIGHT, WIDTH, DEPTH)
        output_shape = list(input_shape)
        output_shape[1:3] = compute_window_slide_size_2d(input_shape[1:3], size, strides, mode)

        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, output_shape, input_op.output.dtype, name)
        self.__input_op = input_op
        self.__weights = weights
        self.__size = size
        self.__strides = strides
        self.__mode = mode

    def run(self):
        ksize = self.__weights.shape[1:3]

        padded_input_values = np.moveaxis(self.__input_op.output.values, 0, -1)
        padded_input_values = apply_padding_2d(padded_input_values, self.__mode, ksize, self.__strides)
        strided_input_values = window_slide_2d(padded_input_values, ksize, self.__strides)
        output_values = np.sum(
            (strided_input_values[None, ...] * self.__weights.values[:, None, None, ..., None]), (3, 4, 5))
        output_values = output_values.swapaxes(0, -1)

        self.output.set_values(output_values)

    def compute_gradients(self, gradients: Variable = None):
        inputs = self.__input_op.output
        batch_size = inputs.shape[0]
        weights = self.__weights.values
        nfilters = weights.shape[0]

        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values

        # Compute operation gradients in respect to input values.
        grads_h, grads_w = grads.shape[1:3]
        _, dilated_grads_indices = dilate_map_2d(grads_w, grads_h, *self.__strides, grads.dtype)
        dilated_grads = np.zeros((batch_size, dilated_grads_indices.size, nfilters), grads.dtype)
        dilated_grads[:, dilated_grads_indices.flat, :] = grads.reshape([batch_size, (grads_h * grads_w), nfilters])
        dilated_grads = dilated_grads.reshape((batch_size, *dilated_grads_indices.shape, nfilters))

        ksize = weights.shape[1:3]
        _dilated_grads_out_size = compute_window_slide_size_2d(dilated_grads.shape[1:3], ksize, 1, 'full')
        _y_input_pad, _x_input_pad = compute_padding_size_2d(inputs.shape[1:3], ksize, self.__strides, self.__mode)
        _padded_input_size = (
            inputs.shape[1] + int(2 * _y_input_pad),
            inputs.shape[2] + int(2 * _x_input_pad))
        if _dilated_grads_out_size != _padded_input_size:
            dilated_grads_h, dilated_grads_w = dilated_grads.shape[1:3]
            padded_dilated_grads = np.zeros(
                [batch_size, dilated_grads_h + 1, dilated_grads_w + 1, nfilters], dilated_grads.dtype)
            padded_dilated_grads[:, :dilated_grads_h, :dilated_grads_w, :] = dilated_grads
            dilated_grads = padded_dilated_grads

        padded_dilated_grads = np.moveaxis(dilated_grads, 0, -1)
        padded_dilated_grads = apply_padding_2d(padded_dilated_grads, 'full', ksize, 1)
        strided_dilated_grads = window_slide_2d(padded_dilated_grads, ksize, 1)
        dx_padded = np.sum(
            (np.moveaxis(strided_dilated_grads, 4, 0)[..., None] * weights[:, None, None, ..., None, :]),
            (0, 3, 4))
        dx = np.moveaxis(strip_padding_2d(dx_padded, (_y_input_pad, _x_input_pad), self.__mode), 2, 0)

        self.grads_dict[inputs] = Variable(dx)


class Dense(Operation):
    def __init__(
            self,
            input_op: Operation,
            units: int,
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = Zeros,
            trainable: bool = True,
            random_state: Union[int, ndarray, Iterable, float, RandomState] = None,
            name: str = None
    ):
        input_shape = input_op.output.shape
        input_type = input_op.output.dtype

        if len(input_shape) != 3:
            raise ValueError('Input layer outputs must have 3 dimensions (BATCH, LENGTH, 1).')
        if input_shape[2] != 1:
            raise ValueError('Input layer outputs 3-rd dimension must be 1 (BATCH, LENGTH, 1).')

        batch_size, n_inputs = input_shape[:2]

        weights_shape = (units, n_inputs)
        bias_shape = (units, 1)

        weights = weights_initializer(weights_shape, input_type).initialize(random_state=random_state)
        bias = bias_initializer(bias_shape, input_type).initialize(random_state=random_state)

        output_shape = (batch_size, units, 1)

        inputs = [input_op]
        var_list = [weights, bias] if trainable else []

        name = self.__class__.__name__ if name is None else name

        super(Dense, self).__init__(inputs, var_list, output_shape, input_type, name)
        self.__input_op = input_op
        self.__weights = weights
        self.__bias = bias
        self.__trainable = trainable

    def run(self):
        inputs = self.__input_op.output
        y = matmul(self.__weights.values[None, ...], inputs.values) + self.__bias.values[None, ...]
        self.output.set_values(y)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        dx = matmul(np.transpose(self.__weights.values[None, ...], (0, 2, 1)), grads)
        dw = matmul(grads, np.transpose(self.__input_op.output.values, (0, 2, 1)))
        db = np.sum(grads, 2, keepdims=True)

        self.grads_dict[self.__input_op.output] = Variable(dx)
        self.grads_dict[self.__weights] = Variable(dw)
        self.grads_dict[self.__bias] = Variable(db)


class SquaredDistance(Operation):
    def __init__(self, x: Operation, y: Operation, name: str = None):
        assert x.output.shape == y.output.shape, 'Logits and labels shapes must match.'
        assert x.output.dtype == y.output.dtype, 'Logits and labels dtypes must match.'

        inputs = [x, y]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super(SquaredDistance, self).__init__(inputs, var_list, x.output.shape, x.output.dtype, name)
        self.__x = x
        self.__y = y

    def run(self):
        flat_x = np.reshape(self.__x.output.values, [np.prod(self.__x.output.shape)])
        flat_y = np.reshape(self.__y.output.values, [np.prod(self.__y.output.shape)])
        flat_outputs = squared_distance(flat_x, flat_y)
        outputs = np.reshape(flat_outputs, self.output.shape)
        self.output.set_values(outputs)

    def compute_gradients(self, gradients: Variable = None):
        batch_size = self.__x.output.shape[0]
        grads = np.ones((batch_size, 1, 1), self.output.dtype) if gradients is None else gradients.values

        flat_grads = np.reshape(grads, [np.prod(grads.shape)])
        x = np.reshape(self.__x.output.values, [np.prod(self.__x.output.shape)])
        y = np.reshape(self.__y.output.values, [np.prod(self.__y.output.shape)])

        flat_dx, flat_dy = squared_distance_prime(x, y, flat_grads)

        dx = np.reshape(flat_dx, self.__x.output.shape)
        dy = np.reshape(flat_dy, self.__y.output.shape)

        self.grads_dict[self.__x.output] = Variable(dx)
        self.grads_dict[self.__y.output] = Variable(dy)


class CrossEntropy(Operation):
    def __init__(self, x: Operation, y: Operation, axis: int = 1, name: str = None):
        assert x.output.shape == y.output.shape, 'Logits and labels shapes must match.'
        assert x.output.dtype == y.output.dtype, 'Logits and labels dtypes must match.'

        output_shape = list(x.output.shape)
        output_shape[axis] = 1

        inputs = [x, y]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super(CrossEntropy, self).__init__(inputs, var_list, output_shape, x.output.dtype, name)
        self.__x = x
        self.__y = y
        self.__axis = axis

    def run(self):
        e = cross_entropy(self.__x.output.values, self.__y.output.values, self.__axis)
        self.output.set_values(e)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        dx, dy = cross_entropy_prime(self.__x.output.values, self.__y.output.values, self.__axis, grads)
        self.grads_dict[self.__x.output] = Variable(dx)
        self.grads_dict[self.__y.output] = Variable(dy)


class ReduceSum(Operation):
    def __init__(self, input_op: Operation, axis: int = 0, keepdim: bool = True, name: str = None):
        input_shape = list(input_op.output.shape)
        if keepdim:
            output_shape = input_shape
            output_shape[axis] = 1
        else:
            input_shape.pop(axis)
            output_shape = input_shape

        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super(ReduceSum, self).__init__(inputs, var_list, output_shape, input_op.output.dtype, name)
        self.__input_op = input_op
        self.__axis = axis
        self.__keepdim = keepdim

    def run(self):
        inputs = self.__input_op.output
        outputs = np.sum(inputs.values, axis=self.__axis, keepdims=self.__keepdim)
        self.output.set_values(outputs)

    def compute_gradients(self, gradients: Variable = None):
        inputs = self.__input_op.output
        if gradients is None:
            dx = np.ones(inputs.shape, inputs.dtype)
        else:
            grads = gradients.values
            if not self.__keepdim:
                grads = np.expand_dims(grads, self.__axis)

            dx = np.repeat(grads, inputs.shape[self.__axis], self.__axis)

        self.grads_dict[self.__input_op.output] = Variable(dx)


class Reshape(Operation):
    def __init__(self, input_op: Operation, new_shape: Sequence[Union[int, None]], name: str = None):
        input_shape = input_op.output.shape
        output_shape = list(new_shape)
        for i in range(len(output_shape)):
            if output_shape[i] is not None:
                continue
            if len(input_shape) > i:
                output_shape[i] = input_shape[i]
            else:
                output_shape[i] = 1

        if np.prod(input_shape) != np.prod(output_shape):
            raise AssertionError('Input and output array\'s number of elements must match.')

        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, output_shape, input_op.output.dtype, name)
        self.__input_op = input_op

    def run(self):
        outputs = np.reshape(self.__input_op.output.values, self.output.shape)
        self.output.set_values(outputs)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        shaped_grads = np.reshape(grads, self.__input_op.output.shape)
        self.grads_dict[self.__input_op.output] = Variable(shaped_grads)


class Flatten(Reshape):
    def __init__(self, input_op: Operation, axis: int = 1, name: str = None):
        input_shape = list(input_op.output.shape)
        output_shape = input_shape[:axis] + ([np.prod(input_shape[axis:])] if axis < len(input_shape) else [])
        if len(output_shape) < 3:
            output_shape += [1] * (3 - len(output_shape))
        name = self.__class__.__name__ if name is None else name
        super().__init__(input_op, output_shape, name)


class Relu(Operation):
    def __init__(self, input_op: Operation, name: str = None):
        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, input_op.output.shape, input_op.output.dtype, name)
        self.__input_op = input_op

    def run(self):
        outputs = relu(self.__input_op.output.values)
        self.output.set_values(outputs)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        dx = relu_prime(self.__input_op.output.values, grads)
        self.grads_dict[self.__input_op.output] = Variable(dx)


class Sigmoid(Operation):
    def __init__(self, input_op: Operation, name: str = None):
        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, input_op.output.shape, input_op.output.dtype, name)
        self.__input_op = input_op

    def run(self):
        outputs = sigmoid(self.__input_op.output.values)
        self.output.set_values(outputs)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        dx = sigmoid_prime(self.__input_op.output.values, grads)
        self.grads_dict[self.__input_op.output] = Variable(dx)


class SoftArgMax(Operation):
    def __init__(self, input_op: Operation, axis: int = 1, name: str = None):
        inputs = [input_op]
        var_list = []

        name = self.__class__.__name__ if name is None else name

        super().__init__(inputs, var_list, input_op.output.shape, input_op.output.dtype, name)
        self.__input_op = input_op
        self.__axis = axis

    def run(self):
        y = softargmax(self.__input_op.output.values, self.__axis)
        self.output.set_values(y)

    def compute_gradients(self, gradients: Variable = None):
        grads = np.ones(self.output.shape, self.output.dtype) if gradients is None else gradients.values
        dx = softargmax_prime(self.__input_op.output.values, self.__axis, grads)
        self.grads_dict[self.__input_op.output] = Variable(dx)


# TODO: Finish the implementation of max pooling layer. Gradients computation
#  for `when ksize==strides` seems not working - reason unknown.

# class MaxPool2D(Operation):
#     def __init__(
#             self,
#             input_layer: Operation,
#             size: Union[int, Tuple[int, int]] = 2,
#             mode: str = 'valid',
#             name: str = None
#     ):
#         if mode not in {'valid', 'same'}:
#             raise ValueError(
#                 'Sliding window mode for MaxPool2D operation can '
#                 'only be one of the fallowing: "valid" or "same".')
#
#         if isinstance(size, int):
#             size = (size, size)
#
#         input_shape = input_layer.outputs.shape  # (BATCH, HEIGHT, WIDTH, DEPTH)
#         output_shape = list(input_shape)
#         output_shape[1:3] = compute_window_slide_size_2d(input_shape[1:3], size, size, mode)
#
#         inputs = [input_layer]
#         var_list = []
#
#         name = self.__class__.__name__ if name is None else name
#
#         super().__init__(inputs, var_list, output_shape, input_layer.outputs.dtype, name)
#         self.__input_layer = input_layer
#         self.__size = size
#         self.__strides = size
#         self.__mode = mode
#
#     def run(self):
#         padded_input_values = np.moveaxis(self.__input_layer.outputs.values, 0, -1)
#         padded_input_values = apply_padding_2d(padded_input_values, self.__size, self.__strides, self.__mode)
#         strided_input_values = window_slide_2d(padded_input_values, self.__size, self.__strides)
#         _ = strided_input_values.reshape((
#             *strided_input_values.shape[:2],
#             np.prod(strided_input_values.shape[2:4]),
#             *strided_input_values.shape[4:]))
#         output_values = np.moveaxis(_.max(2), -1, 0)
#         self.outputs.set_values(output_values)
#
#     def compute_gradients(self, gradients: Variable = None):
#         # inputs = self.__input_layer.outputs
#         # batch_size = inputs.shape[0]
#         # nchannels = inputs.shape[-1]
#         # yksize, xksize = weights.shape[1:3]
#         # ystride, xstride = self.__strides[:2]
#
#         grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
#
#         input_values = self.__input_layer.outputs.values
#         padded_input_values = np.moveaxis(input_values, 0, -1)
#         padded_input_values = apply_padding_2d(padded_input_values, self.__size, self.__strides, self.__mode)
#         strided_input_values = window_slide_2d(padded_input_values, self.__size, self.__strides)
#
#         padded_output_values = np.zeros(padded_input_values.shape, padded_input_values.dtype)
#         strided_output_values = window_slide_2d(padded_output_values, self.__size, self.__strides, False)
#
#         temp_values = strided_output_values.reshape((
#             *strided_input_values.shape[:2],
#             np.prod(strided_input_values.shape[2:4]),
#             *strided_input_values.shape[4:]))
#         temp_values = np.moveaxis(temp_values, 2, -1)
#         temp_values = temp_values.reshape((np.prod(temp_values.shape[:-1]), temp_values.shape[-1]))
#         temp_values[
#             np.arange(temp_values.shape[0]),
#             strided_input_values.reshape(temp_values.shape).argmax(-1)
#         ] = np.moveaxis(grads, 0, -1).flat
#         temp_values = temp_values.reshape(strided_output_values.shape)
#         strided_output_values[:] = temp_values
#
#         output_values = strip_padding_2d(padded_output_values, input_values.shape[1:3], self.__mode)
#         output_values = np.moveaxis(output_values, -1, 0)
#         self.grads_dict[self.__input_layer.outputs] = Variable(output_values)
