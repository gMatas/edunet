import numpy as np
from numpy import ndarray

rand = np.random.RandomState(696969)


def create_algebra_dataset(n: int):
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

print('x:', x.shape)
print('y:', x.shape)

units = 1
w = np.float32(rand.normal(size=[units, 2]))
b = np.zeros([units, 1], np.float32)
print('w:', w.shape)
print('b:', b.shape)

print('\n--- feed-forward ---\n')


def dense_forward(x: ndarray):
    batch_size = x.shape[0]
    y = np.array([np.matmul(w, x[i]) + b for i in range(batch_size)], np.float32)
    return y


dense_out = dense_forward(x)
print('dense_out:', dense_out.shape)


def squared_distance_forward(x: ndarray, y: ndarray):
    assert x.shape == y.shape
    y = (x - y) ** 2.
    return y


sd_out = squared_distance_forward(dense_out, y)
print('sd_out:', sd_out.shape)


def reduce_sum_forward(x: ndarray, axis: int, keepdims: bool):
    if axis == 0:
        y = np.sum(x, axis, keepdims=True)
    else:
        y = np.sum(x, axis, keepdims=keepdims)
    return y


sum_out = reduce_sum_forward(sd_out, 0, True)
print('sum_out:', sum_out.shape)

print('\n--- backprop ---\n')


def reduce_sum_backward(x: ndarray, axis: int, keepdims: bool):
    xx = np.ones(x.shape, x.dtype)
    dx = reduce_sum_forward(xx, axis, keepdims)
    return dx


sum_out_grads = reduce_sum_backward(sd_out, 0, True)
print('sum_out_grads:', sum_out_grads.shape)


def squared_distance_backward(x: ndarray, y: ndarray):
    assert x.shape == y.shape
    dx = (x - y) * 2.
    return dx


sd_out_grads = squared_distance_backward(dense_out, y)
print('sd_out_grads:', sd_out_grads.shape)


def dense_backward(x: ndarray):
    batch_size = x.shape[0]
    dx = np.array([w] * batch_size, w.dtype)
    dw = x
    db = np.array([b] * batch_size, b.dtype)
    return dx, dw, db


dense_out_dx_grads, dense_out_dw_grads, dense_out_db_grads = dense_backward(x)
print('dense_out_dx_grads:', dense_out_dx_grads.shape)
print('dense_out_dw_grads:', dense_out_dw_grads.shape)
print('dense_out_db_grads:', dense_out_db_grads.shape)

print('\n--- chaining ---\n')


def chain_matmul(x1: ndarray, x2: ndarray) -> ndarray:
    x1_batch_size = x1.shape[0]
    x2_batch_size = x2.shape[0]

    if x1_batch_size == x2_batch_size:
        y = np.matmul(x1, x2)
    elif x1_batch_size == 1:
        y = np.array([np.matmul(x1[0], x2[i]) for i in range(x2_batch_size)], np.float32)
    elif x2_batch_size == 1:
        y = np.array([np.matmul(x1[i], x2[0]) for i in range(x1_batch_size)], np.float32)

    return y


o1 = chain_matmul(sum_out_grads, sd_out_grads)
print(o1.shape)
print(dense_out_dw_grads.shape)

# o2 = chain_matmul(o1, dense_out_dw_grads)
# print(o2.shape)
