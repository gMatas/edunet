from typing import Sequence, Union, Tuple
import math

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import as_strided, broadcast_to


__all__ = [
    'isscalar',
    'isvector',
    'ismatrix',
    'repeat',
    'dilate_map_2d',
    'compute_window_slide_size_2d',
    'window_slide_2d',
    'compute_padding_size_2d',
    'apply_padding_2d',
    'strip_padding_2d',
]


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


def repeat(a: ndarray, axis: int, repeats: int, copy: bool = True) -> ndarray:
    if copy:
        output = np.repeat(a, repeats, axis)
        return output

    new_shape = list(a.shape)
    new_shape[axis] = repeats
    output = broadcast_to(a, new_shape)
    return output


def dilate_map_2d(
        xsize: int,
        ysize: int,
        xstride: int,
        ystride: int,
        dtype: Union[type, np.dtype]
) -> Tuple[ndarray, ndarray]:
    ny, nx = ysize, xsize

    mx = xstride * (nx - 1) + 1
    x = (np.arange(0, mx, 1, dtype=np.int64) % xstride == 0)

    my = ystride * (ny - 1) + 1
    y = (np.arange(0, my, 1, dtype=np.int64) % ystride == 0)

    yv, xv = np.meshgrid(y, x)
    yx_indices = (yv & xv)
    yx = np.zeros((my, mx), dtype)

    return yx, yx_indices


def compute_window_slide_size_2d(
        input_size: Tuple[int, int],
        window_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        mode: str
) -> Tuple[int, int]:
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


def window_slide_2d(a: ndarray, size: Union[int, Sequence[int]], strides: Union[int, Sequence[int]] = (1, 1), readonly: bool = True) -> ndarray:
    ysize, xsize = a.shape[:2]  # Input array 2-D size.
    ywinsize, xwinsize = (size, size) if isinstance(size, int) else size[:2]
    ystride, xstride = (strides, strides) if isinstance(strides, int) else strides[:2]

    youtsize, xoutsize = compute_window_slide_size_2d((ysize, xsize), (ywinsize, xwinsize), (ystride, xstride), 'valid')

    assert youtsize > 0 and xoutsize > 0, 'Output dimensions must be greater than zero.'

    output_shape = (youtsize, xoutsize, ywinsize, xwinsize, *(a.shape[2:]))
    input_strides = a.strides
    output_strides = (input_strides[0] * ystride, input_strides[1] * xstride, *input_strides)
    output = as_strided(a, output_shape, output_strides, writeable=(not readonly))

    return output


def compute_padding_size_2d(
        size: Tuple[int, int],
        ksize: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        mode: str
) -> Tuple[float, float]:
    ysize, xsize = size
    yksize, xksize = (ksize, ksize) if isinstance(ksize, int) else ksize[:2]
    ystride, xstride = (strides, strides) if isinstance(strides, int) else strides[:2]

    if mode == 'valid':
        return 0., 0.

    elif mode == 'same':
        x_pad_size = (xstride * (xsize - 1) - xsize + xksize) * .5
        y_pad_size = (ystride * (ysize - 1) - ysize + yksize) * .5
        return y_pad_size, x_pad_size

    elif mode == 'full':
        x_pad_size = (xstride * (math.floor((xsize + xksize - 2) / xstride + 1) - 1) + xksize - xsize) * .5
        y_pad_size = (ystride * (math.floor((ysize + yksize - 2) / ystride + 1) - 1) + yksize - ysize) * .5
        return y_pad_size, x_pad_size

    else:
        raise ValueError(
            'No such padding mode is supported. Available '
            'modes are: valid, same, full (case insensitive).')


def apply_padding_2d(
        a: ndarray,
        mode: str,
        ksize: Union[int, Tuple[int, int]] = None,
        strides: Union[int, Tuple[int, int]] = None,
        padding: Union[float, Tuple[float, float]] = None
) -> ndarray:
    """
    Apply 2-D padding.

    Padding size is calculated in respect to 'valid' window sliding mode.
    """

    ysize, xsize = a.shape[:2]

    if padding is None:
        y_pad_size, x_pad_size = compute_padding_size_2d((ysize, xsize), ksize, strides, mode)
    else:
        y_pad_size, x_pad_size = (padding, padding) if isinstance(padding, int) else padding[:2]

    padded = np.zeros((ysize + int(2 * y_pad_size), xsize + int(2 * x_pad_size), *a.shape[2:]), a.dtype)

    if mode == 'valid':
        return a

    elif mode == 'same':
        y_pad_size_floor = math.floor(y_pad_size)
        x_pad_size_floor = math.floor(x_pad_size)

        padded[y_pad_size_floor:y_pad_size_floor + ysize, x_pad_size_floor:x_pad_size_floor + xsize, ...] = a
        return padded

    elif mode == 'full':
        y_pad_size_ceil = math.ceil(y_pad_size)
        x_pad_size_ceil = math.ceil(x_pad_size)

        padded[y_pad_size_ceil:y_pad_size_ceil + ysize, x_pad_size_ceil:x_pad_size_ceil + xsize, ...] = a
        return padded

    else:
        raise ValueError(
            'No such padding mode is supported. Available '
            'modes are: valid, same, full (case insensitive).')


def strip_padding_2d(a: ndarray, padding_size: Union[float, Tuple[float, float]], mode: str) -> ndarray:
    """
    Strip 2-D padding from padded array.

    Padding size is calculated in respect to 'valid' window sliding mode.
    """

    ysize, xsize = a.shape[:2]
    y_pad_size, x_pad_size = (padding_size, padding_size) if isinstance(padding_size, int) else padding_size[:2]

    ystripedsize = int(-y_pad_size * 2 + ysize)
    xstripedsize = int(-x_pad_size * 2 + xsize)

    if mode == 'valid':
        return a

    elif mode == 'same':
        x_pad_size_floor = math.floor(x_pad_size)
        y_pad_size_floor = math.floor(y_pad_size)

        striped = a[
            y_pad_size_floor:y_pad_size_floor + ystripedsize,
            x_pad_size_floor:x_pad_size_floor + xstripedsize,
            ...
        ]
        return striped

    elif mode == 'full':
        x_pad_size_ceil = math.ceil(x_pad_size)
        y_pad_size_ceil = math.ceil(y_pad_size)

        striped = a[
            y_pad_size_ceil:y_pad_size_ceil + ystripedsize,
            x_pad_size_ceil:x_pad_size_ceil + xstripedsize,
            ...
        ]
        return striped

    else:
        raise ValueError(
            'No such padding mode is supported. Available '
            'modes are: valid, same, full (case insensitive).')


# def convolution2d(src: ndarray, filter: ndarray, xstride: int, ystride: int, mode: str = 'full'):
#     if src.ndim != 3:
#         raise ValueError('Source array must be 3D with shape (Height, Width, Depth).')
#     elif filter.ndim != 2 and (filter.ndim == 3 and filter.shape[2] != src.shape[2]):
#         raise ValueError('Filter array must be 2-D of shape (Height, Width) or 3-D with shape '
#                          '(Height, Width, Depth). In such case filter depth dimension must '
#                          'match that of the source array.')
#
#     mode = mode.lower()
#     if mode not in {'full', 'same', 'valid'}:
#         raise AssertionError('No such padding mode is available. Currently available '
#                              'padding modes are: full | valid | same.')
#
#     h, w, d = src.shape[:3]
#     kernel_height, kernel_width = filter.shape[:2]
#
#     if mode == 'same':
#         x_padded_size = xstride * (w - 1) + kernel_width
#         y_padded_size = ystride * (h - 1) + kernel_height
#
#         x_pad_size = int((x_padded_size - w) * .5)
#         y_pad_size = int((y_padded_size - h) * .5)
#
#         pad = np.zeros((y_padded_size, x_padded_size, d), src.dtype)
#         pad[y_pad_size:h + y_pad_size, x_pad_size:w + x_pad_size, :] = src
#         src = pad
#
#     conv_out = convolve2d(src[:, :, 0], (filter[:, :, 0] if filter.ndim > 2 else filter), mode='full')
#     out = np.empty(list(conv_out.shape)+[d], conv_out.dtype)
#     out[:, :, 0] = conv_out
#
#     if d > 1:
#         for i in range(1, d):
#             conv_out = convolve2d(src[:, :, i], (filter[:, :, i] if filter.ndim > 2 else filter), mode='full')
#             out[:, :, i] = conv_out
#
#         if filter.ndim > 2:
#             out = np.expand_dims(np.sum(out, 2), 2)
#
#     if mode == 'full':
#         out = out[::ystride, ::xstride, :]
#     elif mode == 'valid' or mode == 'same':
#         if np.prod(out.shape) != 1:
#             out = out[
#                 kernel_height - 1:-kernel_height + 1:ystride,
#                 kernel_width - 1:-kernel_width + 1:xstride,
#             ]
#
#     return out
