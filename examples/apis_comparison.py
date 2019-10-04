import time

import numpy as np
from numpy import ndarray

import edunet as net

try:
    import cv2
except ImportError:
    raise ImportError(
        'To run this example script OpenCV library must be installed. '
        'Run shell command `pip install opercv-python` to install OpenCV '
        'python library.')


def create_emoji_dataset_batch(image_1: ndarray, image_2: ndarray, n: int, ratio: float = 0.5, random_state=None):
    assert image_1.dtype == image_2.dtype, 'Data type mismatch.'

    first_half = int(round(n * ratio))
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

    indices = random_state.permutation(n)
    data_batch = data_batch[indices]
    labels_batch = labels_batch[indices]

    return data_batch, labels_batch


image_path_a = r'images/god_damned_frown.bmp'
image_path_b = r'images/god_damned_smile.bmp'

image_a = np.float32(cv2.imread(image_path_a)) / 255.
image_b = np.float32(cv2.imread(image_path_b)) / 255.

assert all(img is not None for img in [image_a, image_b]), 'Images failed to load.'

batch_size = 2
data_type = np.float32

image_a = image_a.astype(data_type)
image_b = image_b.astype(data_type)

SEED = 6969696
LEARNING_RATE = 0.1
N_EPOCH = 200


def low_level_api():
    rng = np.random.RandomState(SEED)

    input_data = net.Input([batch_size, *image_a.shape], data_type)
    input_labels = net.Input([batch_size, 1, 1], data_type)

    conv_1 = net.Convolution2D(input_data, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng)
    relu_1 = net.Relu(conv_1)
    conv_2 = net.Convolution2D(relu_1, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng)
    relu_2 = net.Relu(conv_2)
    conv_3 = net.Convolution2D(relu_2, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng)
    relu_3 = net.Relu(conv_3)
    conv_4 = net.Convolution2D(relu_3, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng)
    relu_4 = net.Relu(conv_4)
    conv_5 = net.Convolution2D(relu_4, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng)
    relu_5 = net.Relu(conv_5)
    flatten = net.Flatten(relu_5)
    dense_1 = net.Dense(flatten, 8, random_state=rng)
    relu_6 = net.Relu(dense_1)
    dense_2 = net.Dense(relu_6, 1, random_state=rng)
    sigmoid_1 = net.Sigmoid(dense_2)
    loss = net.SquaredDistance(sigmoid_1, input_labels)
    reduce_sum = net.ReduceSum(loss, 0)

    for epoch in range(N_EPOCH):

        data_batch, labels_batch = create_emoji_dataset_batch(image_a, image_b, batch_size, 0.5, rng)
        labels_batch = labels_batch[:, 0, :].reshape([batch_size, 1, 1])

        input_data.feed(data_batch)
        input_labels.feed(labels_batch)

        conv_1.run()
        relu_1.run()
        conv_2.run()
        relu_2.run()
        conv_3.run()
        relu_3.run()
        conv_4.run()
        relu_4.run()
        conv_5.run()
        relu_5.run()
        flatten.run()
        dense_1.run()
        relu_6.run()
        dense_2.run()
        sigmoid_1.run()
        loss.run()
        reduce_sum.run()

        reduce_sum.compute_gradients(None)
        loss.compute_gradients(reduce_sum.grads_dict[loss.output])
        sigmoid_1.compute_gradients(gradients=loss.grads_dict[sigmoid_1.output])
        dense_2.compute_gradients(gradients=sigmoid_1.grads_dict[dense_2.output])
        relu_6.compute_gradients(gradients=dense_2.grads_dict[relu_6.output])
        dense_1.compute_gradients(gradients=relu_6.grads_dict[dense_1.output])
        flatten.compute_gradients(gradients=dense_1.grads_dict[flatten.output])
        relu_5.compute_gradients(gradients=flatten.grads_dict[relu_5.output])
        conv_5.compute_gradients(gradients=relu_5.grads_dict[conv_5.output])
        relu_4.compute_gradients(gradients=conv_5.grads_dict[relu_4.output])
        conv_4.compute_gradients(gradients=relu_4.grads_dict[conv_4.output])
        relu_3.compute_gradients(gradients=conv_4.grads_dict[relu_3.output])
        conv_3.compute_gradients(gradients=relu_3.grads_dict[conv_3.output])
        relu_2.compute_gradients(gradients=conv_3.grads_dict[relu_2.output])
        conv_2.compute_gradients(gradients=relu_2.grads_dict[conv_2.output])
        relu_1.compute_gradients(gradients=conv_2.grads_dict[relu_1.output])
        conv_1.compute_gradients(gradients=relu_1.grads_dict[conv_1.output])
        input_labels.compute_gradients()
        input_data.compute_gradients()

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
            str(loss.output.values.mean()),
            str(input_labels.output.values.flatten()),
            str(sigmoid_1.output.values.flatten())
        ))

        for layer in layers:
            for var in layer.var_list:
                if var not in layer.grads_dict:
                    continue
                grads_var = layer.grads_dict[var]
                var.set_values(np.mean(var.values - LEARNING_RATE * grads_var.values, axis=0))


def high_level_api():
    rng = np.random.RandomState(SEED)

    graph = net.Graph()

    input_data = graph.add(net.Input([batch_size, *image_a.shape], data_type))
    input_labels = graph.add(net.Input([batch_size, 1, 1], data_type))
    conv_1 = graph.add(net.Convolution2D(input_data, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng))
    relu_1 = graph.add(net.Relu(conv_1))
    conv_2 = graph.add(net.Convolution2D(relu_1, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng))
    relu_2 = graph.add(net.Relu(conv_2))
    conv_3 = graph.add(net.Convolution2D(relu_2, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng))
    relu_3 = graph.add(net.Relu(conv_3))
    conv_4 = graph.add(net.Convolution2D(relu_3, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng))
    relu_4 = graph.add(net.Relu(conv_4))
    conv_5 = graph.add(net.Convolution2D(relu_4, 4, 3, strides=1, mode='valid', weights_initializer=net.initializers.HeNormal, random_state=rng))
    relu_5 = graph.add(net.Relu(conv_5))
    flatten = graph.add(net.Flatten(relu_5))
    dense_1 = graph.add(net.Dense(flatten, 8, random_state=rng))
    relu_6 = graph.add(net.Relu(dense_1))
    dense_2 = graph.add(net.Dense(relu_6, 1, random_state=rng))
    sigmoid_1 = graph.add(net.Sigmoid(dense_2))
    loss = graph.add(net.SquaredDistance(sigmoid_1, input_labels))
    reduce_sum = graph.add(net.ReduceSum(loss, 0))
    minimize_op = graph.add(net.MomentumOptimizer(LEARNING_RATE, 0.1).minimize(reduce_sum))
    # minimize_op = graph.add(net.GradientDescentOptimizer(LEARNING_RATE).minimize(reduce_sum))

    flow = net.Flow(graph)

    for epoch in range(N_EPOCH):
        data_batch, labels_batch = create_emoji_dataset_batch(image_a, image_b, batch_size, 0.5, rng)
        labels_batch = labels_batch[:, 0, :].reshape([batch_size, 1, 1])

        _, loss_out, sigmoid_1_out, input_labels_out = flow.run(
            [minimize_op, loss, sigmoid_1, input_labels],
            feed_dict={
                input_data: data_batch,
                input_labels: labels_batch
            })

        print('epoch: %d  --  loss: %s  --  labels: %s  --  outs: %s' % (
            epoch,
            str(loss_out.mean()),
            str(input_labels_out.flatten()),
            str(sigmoid_1_out.flatten())
        ))


def timeit(callback: callable) -> float:
    timestamp = time.time()
    callback()
    delta_time = time.time() - timestamp
    return delta_time


# print('\nRunning low-level api...')
# print('Done. Completed in', timeit(low_level_api), 'seconds.')

print('\nRunning high-level api...')
print('Done. Completed in', timeit(high_level_api), 'seconds.')
