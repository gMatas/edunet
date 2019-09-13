from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import as_strided


# n = 8
# a = np.arange(n**2, dtype=np.int64).reshape((n, n))
# print(a[:, :], '\n')
#
# print(a.dtype)     # int64
# print(a.itemsize)  # 8
# print(a.shape)     # (8, 8, 1)
# print(a.strides)   # (64, 8, 8)
# print()
#
# b = as_strided(a, (3, 3, 3, 3), strides=(64*2, 8*2, 64, 8), writeable=False)
# print(b.shape, '\n')
# print(b[:, :, :, :], '\n')
# print(b[-1, 0, :, :], '\n')


def compute_window_slide_size_2d(
        input_size: Tuple[int, int],
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        mode: str
):
    h, w = input_size[:2]
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (strides, strides) if isinstance(strides, int) else strides

    mode = mode.lower()
    if mode == 'valid':
        output_size_h = int((h - kh) / sh + 1)
        output_size_w = int((w - kw) / sw + 1)
    elif mode == 'same':
        output_size_h, output_size_w = h, w
    elif mode == 'full':
        full_size_h = 2 * (kh - 1) + h
        full_size_w = 2 * (kw - 1) + w
        output_size_h = int((full_size_h - kh) / sh + 1)
        output_size_w = int((full_size_w - kw) / sw + 1)
    else:
        raise ValueError('Convolution mode not found.')

    return output_size_h, output_size_w


# n = 8
# a = np.arange(n**2, dtype=np.int64).reshape((n, n, 1)).repeat(2, 2)
# print(a[:, :, 0], '\n')
# print('shape', a.shape)
# print('dtype', a.dtype)
# print('itemsize', a.itemsize)
# print('strides', a.strides)
# print()
#
# b = as_strided(a, (3, 3, 3, 3, 2), strides=(128*2, 16*2, 128, 16, 8), writeable=False)
# print(b.shape, '\n')
# print(b[:, :, :, :, 0], '\n')
# print(b[-1, -1, :, :, 0], '\n')


def window_slide_2d(a: ndarray, size: Sequence[int], strides: Sequence[int] = (1, 1), readonly: bool = True) -> ndarray:
    ysize, xsize = a.shape[:2]  # Input array 2-D size.
    ywinsize, xwinsize = size[:2]  # Window 2-D size.
    ystride, xstride = strides[:2]  # Window slide in 2-D strides sizes.

    youtsize, xoutsize = compute_window_slide_size_2d((ysize, xsize), (ywinsize, xwinsize), (ystride, xstride), 'valid')

    assert youtsize > 0 and xoutsize > 0, 'Output dimensions must be greater than zero.'

    output_shape = (youtsize, xoutsize, ywinsize, xwinsize, *(a.shape[2:]))
    input_strides = a.strides
    output_strides = (input_strides[0] * ystride, input_strides[1] * xstride, *input_strides)
    output = as_strided(a, output_shape, output_strides, writeable=(not readonly))

    return output


n = 8
a = np.arange(n**2, dtype=np.int64).reshape((n, n, 1)).repeat(2, 2)
print(a[:, :, 0], '\n')
print('shape', a.shape)
print('dtype', a.dtype)
print('itemsize', a.itemsize)
print('strides', a.strides)
print()

k = 6
s = 4
b = window_slide_2d(a, (k, k), (s, s))
print('shape', b.shape)
print('strides', b.strides)
print()
print(b[:, :, :, :, 0], '\n')
print(b[0, 0, :, :, 0], '\n')
print(b[-1, -1, :, :, 0], '\n')
