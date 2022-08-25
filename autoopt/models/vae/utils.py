from torch import nn
from math import ceil


def _determine_inverse_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for inverse processes such as transpose convolution
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = in_height * stride_height
    out_width = in_width * stride_width

    # determine the pad size along each dimension
    pad_along_height = max(
        (in_height - 1) * stride_height + kernel_height - out_height, 0
    )
    pad_along_width = max((in_width - 1) * stride_width + kernel_width - out_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def _determine_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0
    )
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'"""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def hook_factory_tf_inverse_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be
    registered on the padding layer to mimic tf's padding 'same' for transpose convolutions."""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_inverse_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def tfconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    tf_padding_type=None,
):
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )
    return nn.Sequential(*modules)


def tfconv2d_transpose(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    tf_padding_type=None,
):
    """Implements tf's padding 'same' for transpose convolutions"""
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_inverse_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    # eliminate the effect of the in-build padding (is not capable of asymmeric padding)
    if isinstance(kernel_size, int):
        padding = kernel_size - 1
    else:
        padding = (kernel_size[0] - 1, kernel_size[1] - 1)

    modules.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
        )
    )

    return nn.Sequential(*modules)
