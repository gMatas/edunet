from typing import Sequence, Tuple, Union
import math

import numpy as np
from numpy import ndarray


def compute_window_slide_size_2d(
        input_size: Tuple[int, int],
        window_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        mode: str
):
    h, w = input_size[:2]
    kh, kw = (window_size, window_size) if isinstance(window_size, int) else window_size
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


def apply_padding_2d(a: ndarray, kernel_size: Sequence[int], strides: Sequence[int], mode: str):
    ysize, xsize = a.shape[:2]
    yksize, xksize = kernel_size[:2]
    ystride, xstride = strides[:2]

    if mode == 'valid':
        return a

    elif mode == 'same':
        x_pad_size = (xstride * (xsize - 1) - xsize + xksize) * .5
        x_pad_size_floor = math.floor(x_pad_size)
        x_pad_size_ceil = math.ceil(x_pad_size)

        y_pad_size = (ystride * (ysize - 1) - ysize + yksize) * .5
        y_pad_size_floor = math.floor(y_pad_size)
        y_pad_size_ceil = math.ceil(y_pad_size)

        padded = np.zeros((
            ysize + y_pad_size_floor + y_pad_size_ceil,
            xsize + x_pad_size_floor + x_pad_size_ceil,
            *(a.shape[2:])
        ), a.dtype)

        padded[y_pad_size_floor:y_pad_size_floor + ysize, x_pad_size_floor:x_pad_size_floor + xsize] = a
        return padded

    elif mode == 'full':
        x_pad_size = xksize - 1
        y_pad_size = yksize - 1

        padded = np.zeros((
            ysize + y_pad_size + y_pad_size,
            xsize + x_pad_size + x_pad_size,
            *(a.shape[2:])
        ), a.dtype)

        padded[y_pad_size:y_pad_size + ysize, x_pad_size:x_pad_size + xsize] = a
        return padded

    else:
        raise ValueError('No such padding mode is found. Available modes are: valid, same, full.')


x = 12
k = 5
s = 2
mode = 'full'

a = np.ones([x, x])
b = apply_padding_2d(a, (k, k), (s, s), mode)
c = compute_window_slide_size_2d(b.shape[:2], (k, k), (s, s), 'valid')

print('input:', a.shape)
print('padded:', b.shape)
print('result:', c)
print('control:', compute_window_slide_size_2d(a.shape[:2], (k, k), (s, s), mode))
# print(b)
