import tensorflow as tf
import numpy as np


def tensor2summary(tensor, tensor_name, scope=None, selected_slices=None):
    with tf.variable_scope(scope):
        if selected_slices is None:
            tf.summary.image(tensor_name, tensor[tf.shape(tensor)[0] // 2, int(tensor.get_shape()[1]) // 2, np.newaxis, :, :, 0, np.newaxis], 1)
        else:
            my_images = tf.transpose(tensor[tf.shape(tensor)[0] // 2,  int(tensor.get_shape()[1]) // 2, np.newaxis, :, :, 0:16], perm=[3, 1, 2, 0])
            tf.summary.image(tensor_name, my_images, max_outputs=16)