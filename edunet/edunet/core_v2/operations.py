import abc
from typing import List, Tuple, Sequence, Union, Optional, Dict, Type, Iterable

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import RandomState

from edunet.core_v2 import Variable
from edunet.core_v2.math import matmul
from edunet.core_v2.math import sigmoid_prime, sigmoid
from edunet.core_v2.math import softmax_prime, softmax
from edunet.core_v2.math import squared_distance, squared_distance_prime
from edunet.core_v2.math import relu, relu_prime
from edunet.core_v2.utilities import compute_window_slide_size_2d, window_slide_2d, apply_padding_2d
from edunet.core_v2.utilities import convolution2d
from edunet.core_v2.utilities import dilate_map_2d
from edunet.core_v2.initializers import Initializer, HeNormal, Zeros


class Operation(abc.ABC):
    def __init__(
            self,
            inputs: list,
            var_list: List[Variable],
            shape: Sequence[int],
            dtype: Union[type, np.dtype]
    ):
        self._inputs: List[Operation] = inputs

        self.outputs: Variable = Variable(None, shape, dtype)
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
    def __init__(self, shape, dtype):
        super(Input, self).__init__(list(), list(), shape, dtype)

    def feed(self, values: ndarray):
        self.outputs.set_values(values)

    def forward(self):
        pass

    def backward(self, gradients: Variable = None):
        pass


class DefaultInput(Input):
    def __init__(self, values: ndarray):
        super().__init__(values.shape, values.dtype)
        self.feed(values)


def feed_pairs(feed_dict: Dict[Input, ndarray]):
    for layer, value in feed_dict.items():
        layer.feed(value)


class Convolution2D(Operation):
    def __init__(
            self,
            input_layer: Operation,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            strides: Union[int, Tuple[int, int]] = 1,
            mode: str = 'valid',
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = Zeros,
            trainable: bool = True,
            random_state: Union[int, ndarray, Iterable, float, RandomState] = None
    ):
        inputs = input_layer.outputs

        input_shape = inputs.shape  # (BATCH, HEIGHT, WIDTH, DEPTH)
        input_type = inputs.dtype
        batch_size = input_shape[0]

        # Initialize kernel and bias tensors.
        kernel_size_y, kernel_size_x = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        kernel_shape = (1, filters, kernel_size_y, kernel_size_x, input_shape[3])
        bias_shape = (1, filters, 1)

        weights = weights_initializer(kernel_shape, input_type).initialize(random_state=random_state)
        bias = bias_initializer(bias_shape, input_type).initialize(random_state=random_state)

        # Set stride parameters.
        stride_y, stride_x = (strides, strides) if isinstance(strides, int) else strides

        mode = mode.lower()
        output_size_y, output_size_x = compute_window_slide_size_2d(
            input_shape[1:3], (kernel_size_y, kernel_size_x), (stride_y, stride_x), mode)
        output_shape = (batch_size, output_size_y, output_size_x, filters)

        inputs = [input_layer]
        var_list = [weights, bias]

        super().__init__(inputs, var_list, output_shape, input_type)
        self.__input_layer = input_layer
        self.__weights = weights
        self.__bias = bias
        self.__strides: Tuple[int, int] = (stride_y, stride_x)
        self.__mode = mode
        self.__trainable = trainable

    def forward(self):
        inputs = self.__input_layer.outputs
        batch_size = inputs.shape[0]
        weights = self.__weights.values[0]
        bias = self.__bias.values[0]
        nfilters = weights.shape[0]
        ystride, xstride = self.__strides[:2]

        outputs = np.empty(self.outputs.shape, self.outputs.dtype)

        for i in range(batch_size):
            input_values = inputs.values[i, :, :, :]
            for j in range(nfilters):
                conv_out = convolution2d(input_values, weights[j, :, :, :], ystride, xstride, self.__mode)
                conv_out = np.squeeze(conv_out, 2) + bias[j, :]
                outputs[i, :, :, j] = conv_out

        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        inputs = self.__input_layer.outputs
        nchannels = inputs.shape[-1]
        batch_size = inputs.shape[0]
        weights = self.__weights.values[0]
        nfilters = weights.shape[0]
        yksize, xksize = weights.shape[1:3]
        ystride, xstride = self.__strides[:2]

        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values

        # Compute operation gradients in respect to input values.
        grads_h, grads_w = grads.shape[1:3]
        _, dilated_grads_indices = dilate_map_2d(grads_w, grads_h, xstride, ystride, grads.dtype)
        dilated_grads = np.zeros((batch_size, dilated_grads_indices.size, nfilters), grads.dtype)
        dilated_grads[:, dilated_grads_indices.flat, :] = grads.reshape([batch_size, (grads_h * grads_w), nfilters])
        dilated_grads = dilated_grads.reshape((batch_size, *dilated_grads_indices.shape, nfilters))

        dxh, dxw = compute_window_slide_size_2d(weights.shape[1:3], dilated_grads.shape[1:3], (1, 1), 'full')
        dx = np.empty(inputs.shape, self.outputs.dtype)
        if (dxh, dxw) != dx.shape[1:3]:
            dilated_grads_h, dilated_grads_w = dilated_grads.shape[1:3]
            padded_dilated_grads = np.zeros([batch_size, dilated_grads_h + 1, dilated_grads_w + 1, nfilters], dilated_grads.dtype)
            padded_dilated_grads[:, :dilated_grads_h, :dilated_grads_w, :] = dilated_grads
            dilated_grads = padded_dilated_grads

        flipped_weights = np.flip(np.moveaxis(weights, 0, 2), [0, 1])
        for i in range(batch_size):
            grads_values = dilated_grads[i, :, :, :]
            for j in range(nchannels):
                dxij = convolution2d(flipped_weights[:, :, :, j], grads_values, 1, 1, mode='full')
                dx[i, :, :, j] = np.squeeze(dxij, 2)

        self.grads_dict[inputs] = Variable(dx)

        # Compute operation gradients in respect to weights values.
        input_values = np.moveaxis(inputs.values, 0, -1)
        input_values = apply_padding_2d(input_values, (yksize, xksize), (ystride, xstride), self.__mode)
        input_values = window_slide_2d(input_values, (yksize, xksize), (ystride, xstride))
        input_values = np.moveaxis(input_values, -1, 0)
        dw = np.flip(np.sum(np.expand_dims(input_values, 3) * grads.reshape((*grads.shape, 1, 1, 1)), (1, 2)), [2, 3])

        self.grads_dict[self.__weights] = Variable(dw)

        # Compute operation gradients in respect to bias values.
        grads_perm = np.moveaxis(grads, -1, 1)
        grads_shaped = np.reshape(grads_perm, (*grads_perm.shape[:2], np.prod(grads_perm.shape[2:])))
        db = np.sum(grads_shaped, 2, keepdims=True)

        self.grads_dict[self.__bias] = Variable(db)


class Dense(Operation):
    def __init__(
            self,
            input_layer: Operation,
            units: int,
            weights_initializer: Type[Initializer] = HeNormal,
            bias_initializer: Type[Initializer] = Zeros,
            trainable: bool = True,
            random_state: Union[int, ndarray, Iterable, float, RandomState] = None
    ):
        input_shape = input_layer.outputs.shape
        input_type = input_layer.outputs.dtype

        if len(input_shape) != 3:
            raise ValueError('Input layer outputs must have 3 dimensions (BATCH, LENGTH, 1).')
        if input_shape[2] != 1:
            raise ValueError('Input layer outputs 3-rd dimension must be 1 (BATCH, LENGTH, 1).')

        batch_size, n_inputs = input_shape[:2]

        weights_shape = (1, units, n_inputs)
        bias_shape = (1, units, 1)

        weights = weights_initializer(weights_shape, input_type).initialize(random_state=random_state)
        bias = bias_initializer(bias_shape, input_type).initialize(random_state=random_state)

        output_shape = (batch_size, units, 1)

        inputs = [input_layer]
        var_list = [weights, bias] if trainable else []

        super(Dense, self).__init__(inputs, var_list, output_shape, input_type)
        self.__input_layer = input_layer
        self.__weights = weights
        self.__bias = bias
        self.__trainable = trainable

    def forward(self):
        inputs = self.__input_layer.outputs
        y = matmul(self.__weights.values, inputs.values) + self.__bias.values
        self.outputs.set_values(y)

    def backward(self, gradients: Variable = None):
        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
        dx = matmul(np.transpose(self.__weights.values, (0, 2, 1)), grads)
        dw = matmul(grads, np.transpose(self.__input_layer.outputs.values, (0, 2, 1)))
        db = np.sum(grads, 2, keepdims=True)

        self.grads_dict[self.__input_layer.outputs] = Variable(dx)
        self.grads_dict[self.__weights] = Variable(dw)
        self.grads_dict[self.__bias] = Variable(db)


class SquaredDistance(Operation):
    def __init__(self, x: Operation, y: Operation):
        assert x.outputs.shape == y.outputs.shape, 'Logits and labels shapes must match.'
        assert x.outputs.dtype == y.outputs.dtype, 'Logits and labels dtypes must match.'

        batch_size = x.outputs.shape[0]
        inputs = [x, y]
        var_list = []

        super(SquaredDistance, self).__init__(inputs, var_list, x.outputs.shape, x.outputs.dtype)
        self.__x = x
        self.__y = y

    def forward(self):
        flat_x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.outputs.shape)])
        flat_y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.outputs.shape)])
        flat_outputs = squared_distance(flat_x, flat_y)
        outputs = np.reshape(flat_outputs, self.outputs.shape)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        batch_size = self.__x.outputs.shape[0]
        grads = np.ones((batch_size, 1, 1), self.outputs.dtype) if gradients is None else gradients.values

        flat_grads = np.reshape(grads, [np.prod(grads.shape)])
        x = np.reshape(self.__x.outputs.values, [np.prod(self.__x.outputs.shape)])
        y = np.reshape(self.__y.outputs.values, [np.prod(self.__y.outputs.shape)])

        flat_dx, flat_dy = squared_distance_prime(x, y, flat_grads)

        dx = np.reshape(flat_dx, self.__x.outputs.shape)
        dy = np.reshape(flat_dy, self.__y.outputs.shape)

        self.grads_dict[self.__x.outputs] = Variable(dx)
        self.grads_dict[self.__y.outputs] = Variable(dy)


class ReduceSum(Operation):
    def __init__(self, input_layer: Operation, axis: int, keepdim: bool = True):
        input_shape = list(input_layer.outputs.shape)
        if keepdim:
            output_shape = input_shape
            output_shape[axis] = 1
        else:
            input_shape.pop(axis)
            output_shape = input_shape

        inputs = [input_layer]
        var_list = []

        super(ReduceSum, self).__init__(inputs, var_list, output_shape, input_layer.outputs.dtype)
        self.__input_layer = input_layer
        self.__axis = axis
        self.__keepdim = keepdim

    def forward(self):
        inputs = self.__input_layer.outputs
        outputs = np.sum(inputs.values, axis=self.__axis, keepdims=self.__keepdim)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        inputs = self.__input_layer.outputs
        if gradients is None:
            dx = np.ones(inputs.shape, inputs.dtype)
        else:
            grads = gradients.values
            if not self.__keepdim:
                grads = np.expand_dims(grads, self.__axis)

            dx = np.repeat(grads, inputs.shape[self.__axis], self.__axis)

        self.grads_dict[self.__input_layer.outputs] = Variable(dx)


class Reshape(Operation):
    def __init__(self, input_layer: Operation, new_shape: Sequence[Union[int, None]]):
        input_shape = input_layer.outputs.shape
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

        inputs = [input_layer]
        var_list = []

        super().__init__(inputs, var_list, output_shape, input_layer.outputs.dtype)
        self.__input_layer = input_layer

    def forward(self):
        outputs = np.reshape(self.__input_layer.outputs.values, self.outputs.shape)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
        shaped_grads = np.reshape(grads, self.__input_layer.outputs.shape)
        self.grads_dict[self.__input_layer.outputs] = Variable(shaped_grads)


class Flatten(Reshape):
    def __init__(self, input_layer: Operation, axis: int = 1):
        input_shape = list(input_layer.outputs.shape)
        output_shape = input_shape[:axis] + ([np.prod(input_shape[axis:])] if axis < len(input_shape) else [])
        if len(output_shape) < 3:
            output_shape += [1] * (3 - len(output_shape))
        super().__init__(input_layer, output_shape)


class Relu(Operation):
    def __init__(self, input_layer: Operation):
        inputs = [input_layer]
        var_list = []

        super().__init__(inputs, var_list, input_layer.outputs.shape, input_layer.outputs.dtype)
        self.__input_layer = input_layer

    def forward(self):
        outputs = relu(self.__input_layer.outputs.values)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
        dx = relu_prime(self.__input_layer.outputs.values, grads)
        self.grads_dict[self.__input_layer.outputs] = Variable(dx)


class Sigmoid(Operation):
    def __init__(self, input_layer: Operation):
        inputs = [input_layer]
        var_list = []

        super().__init__(inputs, var_list, input_layer.outputs.shape, input_layer.outputs.dtype)
        self.__input_layer = input_layer

    def forward(self):
        outputs = sigmoid(self.__input_layer.outputs.values)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
        dx = sigmoid_prime(self.__input_layer.outputs.values, grads)
        self.grads_dict[self.__input_layer.outputs] = Variable(dx)


class SoftArgMax(Operation):
    def __init__(self, input_layer: Operation):
        inputs = [input_layer]
        var_list = []

        super().__init__(inputs, var_list, input_layer.outputs.shape, input_layer.outputs.dtype)
        self.__input_layer = input_layer

    def forward(self):
        outputs = softmax(self.__input_layer.outputs.values)
        self.outputs.set_values(outputs)

    def backward(self, gradients: Variable = None):
        grads = np.ones(self.outputs.shape, self.outputs.dtype) if gradients is None else gradients.values
        dx = softmax_prime(self.__input_layer.outputs.values, grads)
        self.grads_dict[self.__input_layer.outputs] = Variable(dx)


def test_cnn():
    import cv2

    def create_emoji_dataset_batch(image_1: ndarray, image_2: ndarray, n: int, ratio: float = 0.5, random_state=None):
        assert image_1.dtype == image_2.dtype, 'Data type mismatch.'

        first_half = int(round(n * 0.5))
        second_half = n - first_half

        data_batch_1 = np.repeat(np.expand_dims(image_1, 0), first_half, 0)
        data_batch_2 = np.repeat(np.expand_dims(image_2, 0), second_half, 0)
        data_batch = np.concatenate([data_batch_1, data_batch_2], 0)

        dtype = image_1.dtype

        labels_batch_1 = np.zeros((first_half, 2, 1), dtype)
        labels_batch_1[:, 0] = 1
        labels_batch_2 = np.zeros((second_half, 2, 1), dtype)
        labels_batch_2[:, 1] = 1
        labels_batch = np.concatenate([labels_batch_1, labels_batch_2], 0)

        indices = np.random.permutation(n)
        data_batch = data_batch[indices]
        labels_batch = labels_batch[indices]

        return data_batch, labels_batch

    image_path_a = r'D:\My Work\Personal\EduNet\god_damned_smile.bmp'
    image_path_b = r'D:\My Work\Personal\EduNet\god_damned_frown.bmp'

    image_a = np.float32(cv2.imread(image_path_a)) / 255.
    image_b = np.float32(cv2.imread(image_path_b)) / 255.

    batch_size = 2
    data_type = np.float64

    image_a = image_a.astype(data_type)
    image_b = image_b.astype(data_type)

    input_data = Input([batch_size, *image_a.shape], data_type)
    input_labels = Input([batch_size, 1, 1], data_type)

    conv_1 = Convolution2D(input_data, 4, 3, strides=1, mode='valid', weights_initializer=HeNormal)
    relu_1 = Relu(conv_1)
    conv_2 = Convolution2D(relu_1, 4, 3, strides=1, mode='valid', weights_initializer=HeNormal)
    relu_2 = Relu(conv_2)
    conv_3 = Convolution2D(relu_2, 4, 3, strides=1, mode='valid', weights_initializer=HeNormal)
    relu_3 = Relu(conv_3)
    conv_4 = Convolution2D(relu_3, 4, 3, strides=1, mode='valid', weights_initializer=HeNormal)
    relu_4 = Relu(conv_4)
    conv_5 = Convolution2D(relu_4, 4, 3, strides=1, mode='valid', weights_initializer=HeNormal)
    relu_5 = Relu(conv_5)
    flatten = Flatten(relu_5)
    dense_1 = Dense(flatten, 8)
    relu_6 = Relu(dense_1)
    dense_2 = Dense(relu_6, 1)
    sigmoid_1 = Sigmoid(dense_2)
    # softmax_1 = Softmax(dense_2)
    loss = SquaredDistance(sigmoid_1, input_labels)
    reduce_sum = ReduceSum(loss, 0)

    for epoch in range(1000):

        data_batch, labels_batch = create_emoji_dataset_batch(image_a, image_b, batch_size, 0.5)
        labels_batch = labels_batch[:, 0, :].reshape([batch_size, 1, 1])

        input_data.feed(data_batch)
        input_labels.feed(labels_batch)

        conv_1.forward()
        relu_1.forward()
        conv_2.forward()
        relu_2.forward()
        conv_3.forward()
        relu_3.forward()
        conv_4.forward()
        relu_4.forward()
        conv_5.forward()
        relu_5.forward()
        flatten.forward()
        dense_1.forward()
        relu_6.forward()
        dense_2.forward()
        sigmoid_1.forward()
        # softmax_1.forward()
        loss.forward()
        reduce_sum.forward()

        reduce_sum.backward(None)
        loss.backward(reduce_sum.grads_dict[loss.outputs])
        sigmoid_1.backward(gradients=loss.grads_dict[sigmoid_1.outputs])
        # softmax_1.backward(gradients=loss.grads_dict[softmax_1.outputs])
        dense_2.backward(gradients=sigmoid_1.grads_dict[dense_2.outputs])
        relu_6.backward(gradients=dense_2.grads_dict[relu_6.outputs])
        dense_1.backward(gradients=relu_6.grads_dict[dense_1.outputs])
        flatten.backward(gradients=dense_1.grads_dict[flatten.outputs])
        relu_5.backward(gradients=flatten.grads_dict[relu_5.outputs])
        conv_5.backward(gradients=relu_5.grads_dict[conv_5.outputs])
        relu_4.backward(gradients=conv_5.grads_dict[relu_4.outputs])
        conv_4.backward(gradients=relu_4.grads_dict[conv_4.outputs])
        relu_3.backward(gradients=conv_4.grads_dict[relu_3.outputs])
        conv_3.backward(gradients=relu_3.grads_dict[conv_3.outputs])
        relu_2.backward(gradients=conv_3.grads_dict[relu_2.outputs])
        conv_2.backward(gradients=relu_2.grads_dict[conv_2.outputs])
        relu_1.backward(gradients=conv_2.grads_dict[relu_1.outputs])
        conv_1.backward(gradients=relu_1.grads_dict[conv_1.outputs])
        input_labels.backward()
        input_data.backward()

        layers = [
            conv_1,
            relu_1,
            conv_2,
            relu_2,
            conv_3,
            relu_3,
            conv_4,
            relu_4,
            conv_5,
            relu_5,
            flatten,
            dense_1,
            relu_6,
            dense_2,
            loss,
        ]

        print('epoch: %d  --  loss: %s  --  labels: %s  --  outs: %s' % (
            epoch,
            str(loss.outputs.values.mean()),
            str(input_labels.outputs.values.flatten()),
            str(sigmoid_1.outputs.values.flatten())
        ))

        learning_rate = 0.01
        for layer in layers:
            for var in layer.var_list:
                if var not in layer.grads_dict:
                    continue
                grads_var = layer.grads_dict[var]
                var.set_values(np.mean(var.values - learning_rate * grads_var.values, axis=0, keepdims=True))


def test_nn():
    def create_algebra_dataset(n: int, seed: int = None):
        rand = np.random.RandomState(seed)
        x1 = rand.uniform(size=(n, 1))
        x2 = rand.uniform(size=(n, 1))
        y = x1 / 3. + x2 / 7. - 2.
        return x1, x2, y

    batch_size = 4

    input_data = Input([batch_size, 2, 1], np.float64)
    input_labels = Input([batch_size, 1, 1], np.float64)
    dense = Dense(input_data, 1)
    loss = SquaredDistance(dense, input_labels)
    reduce_sum = ReduceSum(loss, 0)

    for i in range(50000):

        X1, X2, Y = create_algebra_dataset(batch_size)
        x = np.reshape([[X1[ii], X2[ii]] for ii in range(batch_size)], [batch_size, 2, 1]).astype(np.float64)
        y = np.reshape([Y[ii] for ii in range(batch_size)], [batch_size, 1, 1]).astype(np.float64)

        feed_pairs({
            input_data: x,
            input_labels: y,
        })

        input_data.forward()
        input_labels.forward()
        dense.forward()
        loss.forward()
        reduce_sum.forward()

        reduce_sum.backward(None)
        loss.backward(reduce_sum.grads_dict[loss.outputs])
        dense.backward(loss.grads_dict[dense.outputs])
        input_labels.backward()
        input_data.backward()

        weights = dense.var_list[0]
        bias = dense.var_list[1]
        print('w1: %f  --  w2: %f  --  b: %f' % (
            weights.values[0, 0, 0],
            weights.values[0, 0, 1],
            bias.values[0, 0, 0],
        ))

        learning_rate = 0.001
        for var in dense.var_list:
            grads_var = dense.grads_dict[var]
            var.set_values(np.mean(var.values - learning_rate * grads_var.values, axis=0, keepdims=True))

    pass


def test_nn_2():
    def create_algebra_dataset(n: int, seed: int = None):
        rand = np.random.RandomState(seed)
        x1 = rand.uniform(size=(n, 1))
        x2 = rand.uniform(size=(n, 1))

        y = x1 / 3. + x2 / 7. - 2
        sigmoid_y = sigmoid_func(y)
        return x1, x2, sigmoid_y

        # y = x1 / 3. - x2 / 7. - 0.05
        # y[y < 0] = 0
        # return x1, x2, y

    batch_size = 4

    input_data = Input([batch_size, 2, 1], np.float64)
    input_labels = Input([batch_size, 1, 1], np.float64)
    dense = Dense(input_data, 1)
    activation = Sigmoid(dense)
    loss = SquaredDistance(activation, input_labels)
    reduce_sum = ReduceSum(loss, 0)

    for i in range(100000):

        X1, X2, Y = create_algebra_dataset(batch_size)
        x = np.reshape([[X1[ii], X2[ii]] for ii in range(batch_size)], [batch_size, 2, 1]).astype(np.float64)
        y = np.reshape([Y[ii] for ii in range(batch_size)], [batch_size, 1, 1]).astype(np.float64)

        feed_pairs({
            input_data: x,
            input_labels: y,
        })

        input_data.forward()
        input_labels.forward()
        dense.forward()
        activation.forward()
        loss.forward()
        reduce_sum.forward()

        reduce_sum.backward(None)
        loss.backward(reduce_sum.grads_dict[loss.outputs])
        activation.backward(loss.grads_dict[activation.outputs])
        dense.backward(activation.grads_dict[dense.outputs])
        input_labels.backward()
        input_data.backward()

        weights = dense.var_list[0]
        bias = dense.var_list[1]
        print('w1: %f  --  w2: %f  --  b: %f  --  y: %s  --  y_hat: %s' % (
            1/weights.values[0, 0, 0],
            1/weights.values[0, 0, 1],
            bias.values[0, 0, 0],
            str(activation.outputs.values.flatten()),
            str(input_labels.outputs.values.flatten()),
        ))

        layers = [
            dense
        ]

        learning_rate = 0.1
        for layer in layers:
            for var in layer.var_list:
                if var not in layer.grads_dict:
                    continue
                grads_var = dense.grads_dict[var]
                var.set_values(np.mean(var.values - learning_rate * grads_var.values, axis=0, keepdims=True))

    pass


def test_nn_3():
    from scipy.special import expit as sigmoid_func

    def create_algebra_dataset(n: int, seed: int = None):
        rand = np.random.RandomState(seed)
        x1 = rand.uniform(size=(n, 1))
        x2 = rand.uniform(size=(n, 1))

        y = x1 / 3. + x2 / 7. - 2
        sigmoid_y = sigmoid_func(y)
        return x1, x2, sigmoid_y

        # y = x1 / 3. - x2 / 7. - 0.05
        # y[y < 0] = 0
        # return x1, x2, y

    batch_size = 4

    input_data = Input([batch_size, 2, 1], np.float64)
    input_labels = Input([batch_size, 1, 1], np.float64)
    reshape_data = Reshape(input_data, [4, 1, 1, 2])
    reshape_labels = Reshape(input_labels, [4, 1, 1, 1])
    conv = Convolution2D(reshape_data, 1, 1)
    activation = Sigmoid(conv)
    loss = SquaredDistance(activation, reshape_labels)
    reduce_sum = ReduceSum(loss, 0)

    for i in range(100000):

        X1, X2, Y = create_algebra_dataset(batch_size)
        x = np.reshape([[X1[ii], X2[ii]] for ii in range(batch_size)], [batch_size, 2, 1]).astype(np.float64)
        y = np.reshape([Y[ii] for ii in range(batch_size)], [batch_size, 1, 1]).astype(np.float64)

        feed_pairs({
            input_data: x,
            input_labels: y,
        })

        input_data.forward()
        input_labels.forward()
        reshape_data.forward()
        reshape_labels.forward()
        conv.forward()
        activation.forward()
        loss.forward()
        reduce_sum.forward()

        reduce_sum.backward(None)
        loss.backward(reduce_sum.grads_dict[loss.outputs])
        activation.backward(loss.grads_dict[activation.outputs])
        conv.backward(activation.grads_dict[conv.outputs])
        reshape_labels.backward(loss.grads_dict[reshape_labels.outputs])
        reshape_data.backward(conv.grads_dict[reshape_data.outputs])
        input_labels.backward()
        input_data.backward()

        weights = conv.var_list[0]
        bias = conv.var_list[1]
        print('w1: %f  --  w2: %f  --  b: %f  --  y: %s  --  y_hat: %s' % (
            1/weights.values[0, 0, 0, 0, 0],
            1/weights.values[0, 0, 0, 0, 1],
            bias.values[0, 0, 0],
            str(activation.outputs.values.flatten()),
            str(input_labels.outputs.values.flatten()),
        ))

        layers = [
            conv
        ]

        learning_rate = 0.1
        for layer in layers:
            for var in layer.var_list:
                if var not in layer.grads_dict:
                    continue
                grads_var = layer.grads_dict[var]
                var.set_values(np.mean(var.values - learning_rate * grads_var.values, axis=0, keepdims=True))

    pass


test_cnn()
# test_nn()
# test_nn_2()
# test_nn_3()
