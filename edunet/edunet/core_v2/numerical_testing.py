import cv2
import numpy as np

import edunet.core_v2.operations as enops
from edunet.core_v2 import Variable


def print_array_info(a):
    print('shape:', a.shape)
    print('dtype:', a.dtype)
    print('mean:', a.mean())
    print('min:', a.min())
    print('max:', a.max())


DTYPE = np.float32
EPSILON = 1e-4

image_path = r'D:\My Work\Personal\EduNet\edunet\lena.png'
original_image = cv2.imread(image_path).astype(DTYPE) / 255.
image = cv2.resize(original_image, (8, 8), interpolation=cv2.INTER_AREA)
data_batch = np.expand_dims(image, 0)
print_array_info(data_batch)
print()

labels_batch = np.ones([1, 1, 1], DTYPE)

seed = 69696969
random_state = np.random.RandomState(seed)


def create_model(layers_dict: dict):
    layers_dict['input_data'] = enops.Input(data_batch.shape, DTYPE)
    layers_dict['input_labels'] = enops.Input(labels_batch.shape, DTYPE)

    layers_dict['conv_1'] = enops.Convolution2D(layers_dict['input_data'], 16, 3, 1, 'valid', random_state=random_state)
    layers_dict['relu_1'] = enops.Relu(layers_dict['conv_1'])
    layers_dict['conv_2'] = enops.Convolution2D(layers_dict['relu_1'], 16, 3, 1, 'valid', random_state=random_state)
    layers_dict['relu_2'] = enops.Relu(layers_dict['conv_2'])
    layers_dict['conv_3'] = enops.Convolution2D(layers_dict['relu_2'], 16, 3, 1, 'valid', random_state=random_state)
    layers_dict['relu_3'] = enops.Relu(layers_dict['conv_3'])
    layers_dict['flatten'] = enops.Flatten(layers_dict['relu_3'])
    layers_dict['dense_1'] = enops.Dense(layers_dict['flatten'], 32, random_state=random_state)
    layers_dict['relu_4'] = enops.Relu(layers_dict['dense_1'])
    layers_dict['dense_2'] = enops.Dense(layers_dict['relu_4'], 1, random_state=random_state)
    layers_dict['sigmoid_activation'] = enops.Sigmoid(layers_dict['dense_2'])

    layers_dict['loss'] = enops.SquaredDistance(layers_dict['sigmoid_activation'], layers_dict['input_labels'])


LAYERS = dict()
create_model(LAYERS)

LAYERS_OTHER = dict()
create_model(LAYERS_OTHER)


def forward_pass(layers_dict: dict):
    layers_dict['input_data'].feed(data_batch)
    layers_dict['input_labels'].feed(labels_batch)

    layers_dict['input_data'].forward()
    layers_dict['input_labels'].forward()
    layers_dict['conv_1'].forward()
    layers_dict['relu_1'].forward()
    layers_dict['conv_2'].forward()
    layers_dict['relu_2'].forward()
    layers_dict['conv_3'].forward()
    layers_dict['relu_3'].forward()
    layers_dict['flatten'].forward()
    layers_dict['dense_1'].forward()
    layers_dict['relu_4'].forward()
    layers_dict['dense_2'].forward()
    layers_dict['sigmoid_activation'].forward()
    layers_dict['loss'].forward()


def backward_pass(layers_dict: dict):
    layers_dict['loss'].backward(None)
    layers_dict['sigmoid_activation'].backward(layers_dict['loss'].grads_dict[layers_dict['sigmoid_activation'].outputs])
    layers_dict['dense_2'].backward(layers_dict['sigmoid_activation'].grads_dict[layers_dict['dense_2'].outputs])
    layers_dict['relu_4'].backward(layers_dict['dense_2'].grads_dict[layers_dict['relu_4'].outputs])
    layers_dict['dense_1'].backward(layers_dict['relu_4'].grads_dict[layers_dict['dense_1'].outputs])
    layers_dict['flatten'].backward(layers_dict['dense_1'].grads_dict[layers_dict['flatten'].outputs])
    layers_dict['relu_3'].backward(layers_dict['flatten'].grads_dict[layers_dict['relu_3'].outputs])
    layers_dict['conv_3'].backward(layers_dict['relu_3'].grads_dict[layers_dict['conv_3'].outputs])
    layers_dict['relu_2'].backward(layers_dict['conv_3'].grads_dict[layers_dict['relu_2'].outputs])
    layers_dict['conv_2'].backward(layers_dict['relu_2'].grads_dict[layers_dict['conv_2'].outputs])
    layers_dict['relu_1'].backward(layers_dict['conv_2'].grads_dict[layers_dict['relu_1'].outputs])
    layers_dict['conv_1'].backward(layers_dict['relu_1'].grads_dict[layers_dict['conv_1'].outputs])
    layers_dict['input_labels'].backward(None)
    layers_dict['input_data'].backward(None)


def compute_explicit_gradients(variable: Variable) -> np.ndarray:
    num_grads = np.empty(variable.shape, variable.dtype)
    for i in range(variable.values.size):
        base_value = variable.values.flat[i]

        variable.values.flat[i] = base_value - EPSILON
        forward_pass(LAYERS)
        loss_1 = LAYERS['loss'].outputs.values

        variable.values.flat[i] = base_value + EPSILON
        forward_pass(LAYERS)
        loss_2 = LAYERS['loss'].outputs.values

        variable.values.flat[i] = base_value
        num_grads.flat[i] = (loss_2 - loss_1) / (2 * EPSILON)

    return num_grads


variables_1 = [
    *LAYERS['conv_1'].var_list,
    *LAYERS['conv_2'].var_list,
    *LAYERS['conv_3'].var_list,
    *LAYERS['dense_1'].var_list,
    *LAYERS['dense_2'].var_list,
]

variables_2 = [
    *LAYERS_OTHER['conv_1'].var_list,
    *LAYERS_OTHER['conv_2'].var_list,
    *LAYERS_OTHER['conv_3'].var_list,
    *LAYERS_OTHER['dense_1'].var_list,
    *LAYERS_OTHER['dense_2'].var_list,
]

forward_pass(LAYERS_OTHER)
backward_pass(LAYERS_OTHER)

for i, (var1, var2) in enumerate(zip(variables_1, variables_2)):
    var_num_grads_1 = compute_explicit_gradients(var1)
    var_num_grads_2 = var2.values
    var_grads_error = np.abs(var_num_grads_2 - var_num_grads_1)
    print('i: {} -- min: {} -- max: {}'.format(i+1, var_grads_error.min(), var_grads_error.max()))



