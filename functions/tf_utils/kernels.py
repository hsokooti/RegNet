import numpy as np
import functions.kernel.conv_kernel as convKernel
import tensorflow as tf


def kernel_bspline_r4():
    kernel_r4 = convKernel.convDownsampleKernel('bspline', 3, 15, normalizeKernel=1)
    kernel_r4 = np.expand_dims(kernel_r4, -1)
    kernel_r4 = np.expand_dims(kernel_r4, -1)
    kernel_r4 = tf.constant(kernel_r4)
    return kernel_r4


def kernel_bspline_r2():
    kernel_r2 = convKernel.convDownsampleKernel('bspline', 3, 7, normalizeKernel=1)
    kernel_r2 = np.expand_dims(kernel_r2, -1)
    kernel_r2 = np.expand_dims(kernel_r2, -1)
    kernel_r2 = tf.constant(kernel_r2)
    return kernel_r2
