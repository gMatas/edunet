module Utilities

export
convolution2d,
compute_convolution_output_size_2d,
CONV_FULL,
CONV_VALID,
CONV_SAME

using DSP

const CONV_FULL = "full"
const CONV_VALID = "valid"
const CONV_SAME = "same"

function compute_convolution_output_size_2d(
        input_size::Tuple{Integer, Integer},
        kernel_size::Union{Integer, Tuple{Integer, Integer}},
        strides::Union{Integer, Tuple{Integer, Integer}},
        mode:: AbstractString = "valid"
    )
    h, w = input_size[1:2]
    kh, kw = isa(kernel_size, Integer) ? (kernel_size, kernel_size) : kernel_size
    sh, sw = isa(strides, Integer) ? (strides, strides) : strides

    mode = lowercase(mode)
    if mode == "valid"
        output_size_h = floor(Int64, (h - kh) / sh + 1)
        output_size_w = floor(Int64, (w - kw) / sw + 1)
    elseif mode == "same"
        output_size_h, output_size_w = input_size[1:2]
    elseif mode == "full"
        full_size_h = 2 * (kh - 1) + h
        full_size_w = 2 * (kw - 1) + w
        output_size_h = floor(Int64, (full_size_h - kh) / sh + 1)
        output_size_w = floor(Int64, (full_size_w - kw) / sw + 1)
    else
        error("2-D convolution mode is not found.")
    end

    return output_size_h, output_size_w
end

function convolution2d(src, filter; x_stride::Int=1, y_stride::Int=1, mode::String=CONV_FULL)
    if ndims(src) != 3
        error("Source array must be 3D with shape (Height x Width x Depth).")
    elseif ndims(filter) != 3 || size(filter, 3) != size(src, 3)
        error("Filter array must be 3D with shape (Height x Width x Depth) " *
            "and its depth size must be same as the source array depth.")
    end

    mode = lowercase(mode)
    if !(mode in [CONV_FULL, CONV_VALID, CONV_SAME])
        error("No such padding mode is available. Currently available " *
            "padding modes are: full | valid | same.")
    end

    h, w, d = size(src)[1:3]
    kernel_height, kernel_width = size(filter)[1:2]

    if mode == CONV_SAME
        x_padded_size = x_stride * (w - 1) + kernel_width
        y_padded_size = y_stride * (h - 1) + kernel_height

        x_pad_size = floor(Int, (x_padded_size - w) * .5)
        y_pad_size = floor(Int, (y_padded_size - h) * .5)

        pad = zeros(eltype(src), y_padded_size, x_padded_size, d)
        pad[1+y_pad_size:h+y_pad_size, 1+x_pad_size:w+x_pad_size, :] = src
        src = pad
    end

    conv_out = DSP.conv2(src[:,:,1], filter[:,:,1])
    out = Array{eltype(conv_out), 3}(undef, size(conv_out)..., d)
    out[:, :, 1] = conv_out

    if d > 1
        for i = 2:d
            conv_out = DSP.conv2(src[:,:,i], filter[:,:,i])
            out[:,:,i] = conv_out
        end

        out = sum(out, dims=3)
    end

    if mode == CONV_FULL
        out = out[1:y_stride:end, 1:x_stride:end, :]
    elseif mode == CONV_VALID || mode == CONV_SAME
        out = out[
            kernel_height:y_stride:end-kernel_height+1,
            kernel_width:x_stride:end-kernel_width+1, :]
    end

    return out
end

end
