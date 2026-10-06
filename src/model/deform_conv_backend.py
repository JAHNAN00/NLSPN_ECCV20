"""Lazy, explicit DCN backend; torchvision inference needs no local extension."""

from torchvision.ops import deform_conv2d


def modulated_deform_conv2d(input, offset, mask, weight, bias, stride=1,
                          padding=0, dilation=1, groups=1, deformable_groups=1,
                          im2col_step=64, backend="torchvision"):
    if backend == "torchvision":
        return deform_conv2d(input, offset, weight, bias, stride=stride,
                             padding=padding, dilation=dilation, mask=mask)
    if backend == "legacy_dcn":
        from .modulated_deform_conv_func import ModulatedDeformConvFunction
        return ModulatedDeformConvFunction.apply(
            input, offset, mask, weight, bias, stride, padding, dilation,
            groups, deformable_groups, im2col_step)
    raise ValueError(f"Unsupported DCN backend: {backend}")
