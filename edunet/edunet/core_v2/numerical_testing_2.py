from typing import Dict

import cv2
import numpy as np

import edunet.core_v2.operations as enops
from edunet.core_v2.operations import Operation
from edunet.core_v2 import Variable


def print_array_info(a):
    print('shape:', a.shape)
    print('dtype:', a.dtype)
    print('mean:', a.mean())
    print('min:', a.min())
    print('max:', a.max())


DTYPE = np.float64
EPSILON = 1e-6
SEED = 69696969


image_path = r'D:\My Work\Personal\EduNet\edunet\lena.png'
original_image = cv2.imread(image_path).astype(DTYPE) / 255.
image = cv2.resize(original_image, (8, 8), interpolation=cv2.INTER_AREA)
data_batch = np.expand_dims(image, 0)
# print_array_info(image)
# data_batch = image.swapaxes(0, -1).reshape((48, 1, 1))
# data_batch = image.swapaxes(0, -1).reshape((3, 16, 1))
data_batch.flat[np.random.RandomState(SEED).choice(range(0, data_batch.size), int(data_batch.size/2))] *= -1
print_array_info(data_batch)
print()

labels_batch = np.ones([1, 1, 1], DTYPE)


def build_model(layers: Dict[str, Operation]):
    random_state = np.random.RandomState(SEED)

    layers['input_data'] = enops.Input([1, 8, 8, 3], DTYPE)
    layers['input_labels'] = enops.Input([1, 1, 1], DTYPE)

    layers['conv_1'] = enops.Convolution2D(layers['input_data'], 4, 3, 1, 'valid', random_state=random_state)
    layers['relu_1'] = enops.Relu(layers['conv_1'])
    layers['conv_2'] = enops.Convolution2D(layers['relu_1'], 4, 3, 2, 'valid', random_state=random_state)
    layers['relu_2'] = enops.Relu(layers['conv_2'])
    layers['flatten'] = enops.Flatten(layers['relu_2'])
    layers['dense_1'] = enops.Dense(layers['flatten'], 5, random_state=random_state)
    layers['relu'] = enops.Relu(layers['dense_1'])
    layers['dense_2'] = enops.Dense(layers['relu'], 1, random_state=random_state)
    layers['sigmoid'] = enops.Sigmoid(layers['dense_2'])
    layers['loss'] = enops.SquaredDistance(layers['sigmoid'], layers['input_labels'])
    layers['reduce_sum'] = enops.ReduceSum(layers['loss'], 0)


explicit_model: Dict[str, Operation] = dict()
build_model(explicit_model)

implicit_model: Dict[str, Operation] = dict()
build_model(implicit_model)


def forward_pass(layers: Dict[str, Operation]):
    layers['input_data'].feed(data_batch)
    layers['input_labels'].feed(labels_batch)

    layers['input_data'].forward()
    layers['input_labels'].forward()
    layers['conv_1'].forward()
    layers['relu_1'].forward()
    layers['conv_2'].forward()
    layers['relu_2'].forward()
    layers['flatten'].forward()
    layers['dense_1'].forward()
    layers['relu'].forward()
    layers['dense_2'].forward()
    layers['sigmoid'].forward()
    layers['loss'].forward()
    layers['reduce_sum'].forward()


def backward_pass(layers: Dict[str, Operation]):
    layers['reduce_sum'].backward()
    layers['loss'].backward(layers['reduce_sum'].grads_dict[layers['loss'].outputs])
    layers['sigmoid'].backward(layers['loss'].grads_dict[layers['sigmoid'].outputs])
    layers['dense_2'].backward(layers['sigmoid'].grads_dict[layers['dense_2'].outputs])
    layers['relu'].backward(layers['dense_2'].grads_dict[layers['relu'].outputs])
    layers['dense_1'].backward(layers['relu'].grads_dict[layers['dense_1'].outputs])
    layers['flatten'].backward(layers['dense_1'].grads_dict[layers['flatten'].outputs])

    layers['relu_2'].backward(layers['flatten'].grads_dict[layers['relu_2'].outputs])
    layers['conv_2'].backward(layers['relu_2'].grads_dict[layers['conv_2'].outputs])
    layers['relu_1'].backward(layers['conv_2'].grads_dict[layers['relu_1'].outputs])

    layers['conv_1'].backward(layers['relu_1'].grads_dict[layers['conv_1'].outputs])
    layers['input_labels'].backward()
    layers['input_data'].backward()


def compute_explicit_gradients(layers: Dict[str, Operation], final_op_name: str, variable: Variable) -> np.ndarray:
    num_grads = np.empty(variable.shape, variable.dtype)
    for i in range(variable.values.size):
        base_value = variable.values.flat[i]

        variable.values.flat[i] = base_value - EPSILON
        forward_pass(layers)
        loss_1 = layers[final_op_name].outputs.values

        variable.values.flat[i] = base_value + EPSILON
        forward_pass(layers)
        loss_2 = layers[final_op_name].outputs.values

        variable.values.flat[i] = base_value
        num_grads.flat[i] = (loss_2 - loss_1) / (2 * EPSILON)

    return num_grads


# # Var list numerical validation.
# forward_pass(explicit_model)
# variable = explicit_model['conv_1'].var_list[1]
# num_grads = compute_explicit_gradients(explicit_model, 'reduce_sum', variable)
#
# forward_pass(implicit_model)
# backward_pass(implicit_model)
# grads = implicit_model['conv_1'].grads_dict[implicit_model['conv_1'].var_list[1]].values

# Layer output variables validation.
forward_pass(explicit_model)
variable = explicit_model['input_data'].outputs
num_grads = compute_explicit_gradients(explicit_model, 'reduce_sum', variable)

forward_pass(implicit_model)
backward_pass(implicit_model)
grads = implicit_model['conv_1'].grads_dict[implicit_model['input_data'].outputs].values

print(num_grads.shape)
print(grads.shape)
print()

grads_error = np.abs(num_grads - grads)

print(grads_error.flatten())
print()
print(np.all(grads_error < EPSILON))
print()
