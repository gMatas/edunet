from typing import Dict

import cv2
import numpy as np

import edunet as net
from edunet import Operation, Variable


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
image = cv2.resize(original_image, (6, 6), interpolation=cv2.INTER_AREA)
data_batch = np.expand_dims(image, 0)
data_batch.flat[np.random.RandomState(SEED).choice(range(0, data_batch.size), int(data_batch.size/2))] *= -1
print_array_info(data_batch)
print()


labels_batch = np.random.RandomState(SEED).uniform(0, 1, [1, 3, 1]).astype(DTYPE)


def build_model(layers: Dict[str, Operation]):
    random_state = np.random.RandomState(SEED)

    layers['input_data'] = net.Input([1, 6, 6, 3], DTYPE)
    layers['input_labels'] = net.Input([1, 3, 1], DTYPE)

    layers['conv_1'] = net.Convolution2D(layers['input_data'], 5, 2, strides=3, mode='valid', random_state=random_state)
    layers['relu_1'] = net.Relu(layers['conv_1'])
    layers['pool_1'] = net.AveragePool2D(layers['relu_1'], 2, mode='valid')
    layers['flatten'] = net.Flatten(layers['pool_1'])
    layers['dense_1'] = net.Dense(layers['flatten'], 5, random_state=random_state)
    layers['relu'] = net.Relu(layers['dense_1'])
    layers['dense_2'] = net.Dense(layers['relu'], 3, random_state=random_state)
    layers['softmax'] = net.SoftArgMax(layers['dense_2'], 1)
    layers['loss'] = net.CrossEntropy(layers['softmax'], layers['input_labels'], 1)
    layers['reduce_sum'] = net.ReduceSum(layers['loss'], 0)


explicit_model: Dict[str, Operation] = dict()
build_model(explicit_model)

implicit_model: Dict[str, Operation] = dict()
build_model(implicit_model)


def forward_pass(layers: Dict[str, Operation]):
    layers['input_data'].feed(data_batch)
    layers['input_labels'].feed(labels_batch)

    layers['input_data'].run()
    layers['input_labels'].run()
    layers['conv_1'].run()
    layers['relu_1'].run()
    layers['pool_1'].run()
    layers['flatten'].run()
    layers['dense_1'].run()
    layers['relu'].run()
    layers['dense_2'].run()
    layers['softmax'].run()
    layers['loss'].run()
    layers['reduce_sum'].run()


def backward_pass(layers: Dict[str, Operation]):
    layers['reduce_sum'].compute_gradients()
    layers['loss'].compute_gradients(layers['reduce_sum'].grads_dict[layers['loss'].output])
    layers['softmax'].compute_gradients(layers['loss'].grads_dict[layers['softmax'].output])
    layers['dense_2'].compute_gradients(layers['softmax'].grads_dict[layers['dense_2'].output])
    layers['relu'].compute_gradients(layers['dense_2'].grads_dict[layers['relu'].output])
    layers['dense_1'].compute_gradients(layers['relu'].grads_dict[layers['dense_1'].output])
    layers['flatten'].compute_gradients(layers['dense_1'].grads_dict[layers['flatten'].output])
    layers['pool_1'].compute_gradients(layers['flatten'].grads_dict[layers['pool_1'].output])
    layers['relu_1'].compute_gradients(layers['pool_1'].grads_dict[layers['relu_1'].output])
    layers['conv_1'].compute_gradients(layers['relu_1'].grads_dict[layers['conv_1'].output])

    layers['input_labels'].compute_gradients()
    layers['input_data'].compute_gradients()


def compute_explicit_gradients(layers: Dict[str, Operation], final_op_name: str, variable: Variable) -> np.ndarray:
    num_grads = np.empty(variable.shape, variable.dtype)
    for i in range(variable.values.size):
        base_value = variable.values.flat[i]

        variable.values.flat[i] = base_value - EPSILON
        forward_pass(layers)
        loss_1 = layers[final_op_name].output.values

        variable.values.flat[i] = base_value + EPSILON
        forward_pass(layers)
        loss_2 = layers[final_op_name].output.values

        variable.values.flat[i] = base_value
        num_grads.flat[i] = (loss_2 - loss_1) / (2 * EPSILON)

    return num_grads


# # Var list numerical validation.
# i_var = 1
# forward_pass(explicit_model)
# variable = explicit_model['conv_1'].var_list[i_var]
# num_grads = compute_explicit_gradients(explicit_model, 'reduce_sum', variable)
#
# forward_pass(implicit_model)
# backward_pass(implicit_model)
# grads = implicit_model['conv_1'].grads_dict[implicit_model['conv_1'].var_list[i_var]].values

# Layer output variables validation.
forward_pass(explicit_model)
variable = explicit_model['input_data'].output
num_grads = compute_explicit_gradients(explicit_model, 'reduce_sum', variable)

forward_pass(implicit_model)
backward_pass(implicit_model)
grads = implicit_model['conv_1'].grads_dict[implicit_model['input_data'].output].values

print(num_grads.shape)
print(grads.shape)
print()

grads_error = np.abs(num_grads.ravel() - grads.ravel())

print(num_grads.flatten())
print()
print(grads.flatten())
print()

print(grads_error)
print()
print(np.all(grads_error < EPSILON))
print()
