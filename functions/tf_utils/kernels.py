import numpy as np
import functions.convKernel as convKernel
import tensorflow as tf


def kernel_bspline_s4():
    kernel_s4 = convKernel.convDownsampleKernel('bspline', 3, 15, normalizeKernel=1)
    kernel_s4 = np.expand_dims(kernel_s4, -1)
    kernel_s4 = np.expand_dims(kernel_s4, -1)
    kernel_s4 = tf.constant(kernel_s4)
    return kernel_s4


def kernel_bspline_s2():
    kernel_s2 = convKernel.convDownsampleKernel('bspline', 3, 7, normalizeKernel=1)
    kernel_s2 = np.expand_dims(kernel_s2, -1)
    kernel_s2 = np.expand_dims(kernel_s2, -1)
    kernel_s2 = tf.constant(kernel_s2)
    return kernel_s2
