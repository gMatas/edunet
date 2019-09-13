from typing import Sequence, Union, Tuple, Iterable

import numpy as np
from numpy import ndarray
from scipy.signal import convolve2d


def isscalar(a: ndarray) -> bool:
    if len(a.shape) > 0:
        return max(a.shape) == 1
    return True


def isvector(a: ndarray):
    if a.ndim == 2:
        return min(a.shape) == 1
    else:
        return a.ndim == 1


def ismatrix(a: ndarray):
    return a.ndim == 2


def compute_convout_size_2d(
        input_size: Sequence[int],
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


def convolution2d(src: ndarray, filter: ndarray, xstride: int, ystride: int, mode: str = 'full'):
    if src.ndim != 3:
        raise AssertionError('Source array must be 3D with shape (Height x Width x Depth).')
    elif filter.ndim != 3 or filter.shape[2] != src.shape[2]:
        raise AssertionError('Source array must be 3D with shape (Height x Width x Depth).')

    mode = mode.lower()
    if mode not in {'full', 'same', 'valid'}:
        raise AssertionError('No such padding mode is available. Currently available '
                             'padding modes are: full | valid | same.')

    h, w, d = src.shape[:3]
    kernel_height, kernel_width = filter.shape[:2]

    if mode == 'same':
        x_padded_size = xstride * (w - 1) + kernel_width
        y_padded_size = ystride * (h - 1) + kernel_height

        x_pad_size = int((x_padded_size - w) * .5)
        y_pad_size = int((y_padded_size - h) * .5)

        pad = np.zeros((y_padded_size, x_padded_size, d), src.dtype)
        pad[y_pad_size:h + y_pad_size, x_pad_size:w + x_pad_size, :] = src
        src = pad

    conv_out = convolve2d(src[:, :, 0], filter[:, :, 0], mode='full')
    out = np.empty(list(conv_out.shape)+[d], conv_out.dtype)
    out[:, :, 0] = conv_out

    if d > 1:
        for i in range(1, d):
            conv_out = convolve2d(src[:, :, i], filter[:, :, i], mode='full')
            out[:, :, i] = conv_out

        out = np.expand_dims(np.sum(out, 2), 2)

    if mode == 'full':
        out = out[::ystride, ::xstride, :]
    elif mode == 'valid' or mode == 'same':
        out = out[
            kernel_height - 1:-kernel_height + 1:ystride,
            kernel_width - 1:-kernel_width + 1:xstride,
        ]

    return out
