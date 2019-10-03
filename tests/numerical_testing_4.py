from typing import Dict

import numpy as np

import edunet as net
from edunet.core import Operation
from edunet.core import Variable


EPSILON = 1e-6
SEED = 69696969
RANDOM_STATE = np.random.RandomState(SEED)

INPUT_DTYPE = np.float64
INPUT_DATA_SHAPE = (1, 1, 50)
INPUT_LABELS_SHAPE = (1, 1, 50)


data_batch = RANDOM_STATE.uniform(0, 1, INPUT_DATA_SHAPE).astype(INPUT_DTYPE)
labels_batch = RANDOM_STATE.uniform(0, 1, INPUT_LABELS_SHAPE).astype(INPUT_DTYPE)


def build_model(layers: Dict[str, Operation]):
    random_state = np.random.RandomState(SEED)

    layers['input_data'] = net.Input(INPUT_DATA_SHAPE, INPUT_DTYPE)
    layers['input_labels'] = net.Input(INPUT_LABELS_SHAPE, INPUT_DTYPE)

    layers['ce'] = net.SoftargmaxCrossEntropyWithLogits(layers['input_labels'], layers['input_data'], -1)


explicit_model: Dict[str, Operation] = dict()
build_model(explicit_model)

implicit_model: Dict[str, Operation] = dict()
build_model(implicit_model)


def forward_pass(layers: Dict[str, Operation]):
    layers['input_data'].feed(data_batch)
    layers['input_labels'].feed(labels_batch)

    layers['input_data'].run()
    layers['input_labels'].run()
    layers['ce'].run()


def backward_pass(layers: Dict[str, Operation], final_op_name: str, op_name: str):
    layers['gradients'] = net.Gradients(layers[final_op_name], [layers[op_name]])
    layers['gradients'].run()


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
num_grads = compute_explicit_gradients(explicit_model, 'ce', variable)

forward_pass(implicit_model)
# backward_pass(implicit_model, 'ce', 'input_labels')
gradients = net.Gradients(implicit_model['ce'], None)
gradients.run()
print(gradients.output.values[0][implicit_model['input_data'].output].values)

# grads = implicit_model['gradients'].output.values[0]

# print(grads)


# print('shapes:')
# print(num_grads.shape)
# print(grads.shape)
# print()
#
# grads_error = np.abs(num_grads.ravel() - grads.ravel())
#
#
# print('flattened num_grads:\n', num_grads.ravel(), '\n')
# print('flattened grads:\n', grads.ravel(), '\n')
#
# print('grads error:\n', grads_error, '\n')
# print('is analytical gradients correct?', np.all(grads_error < EPSILON), '\n')
