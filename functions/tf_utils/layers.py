import tensorflow as tf
import numpy as np
import functions.convKernel as convKernel


def conv3d(input_layer, filters, kernel_size, padding='valid', bn_training=None, dilation_rate=(1, 1, 1), scope=None, activation=None):
    """
    :param input_layer:
    :param filters:
    :param kernel_size:
    :param padding: 'valid' or 'same'
    :param bn_training: None (default): not batch_normalization, 1: batch normalization in training mode, 0: batch _normalization in test mode:
    :param dilation_rate:
    :param scope:
    :param activation:
    :return:
    """
    with tf.variable_scope(scope):
        net = tf.layers.conv3d(input_layer, filters, kernel_size, padding=padding, dilation_rate=dilation_rate)
        if bn_training is not None:
            net = tf.layers.batch_normalization(net, training=bn_training)
        if activation is not None:
            if activation == 'LReLu':
                net = tf.nn.leaky_relu(net)
            elif activation == 'ReLu':
                net = tf.nn.relu(net)
            elif activation == 'ELu':
                net = tf.nn.elu(net)
            else:
                raise ValueError('activation=' + activation + ' is not defined in tfu.conv3d. Valid options: "ReLu", "LReLu"')
    return net


def conv3d_transpose(input_layer, filters, kernel_size, padding='valid', bn_training=None, strides=(1, 1, 1),
                     scope=None, activation=None, use_bias=False, initializer=None, trainable=True):
    """

    :param input_layer:
    :param filters:
    :param kernel_size:
    :param padding:
    :param bn_training: None: not batch_normalization, 1: batch normalization in training mode, 0: batch _normalization in test mode:
    :param strides:
    :param scope:
    :param activation:
    :param use_bias:
    :param initializer: None (default) or 'trilinear'
    :param trainable:
    :return:
    """

    kernel_initializer = None
    if initializer is not None:
        if initializer == 'trilinear':
            conv_kernel_trilinear = convKernel.bilinearUpKernel(dim=3)
            conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
            conv_kernel_trilinear = np.repeat(conv_kernel_trilinear, filters, axis=-1)
            conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
            conv_kernel_trilinear = np.repeat(conv_kernel_trilinear, int(input_layer.get_shape()[4]), axis=-1)
            # size of the conv_kernel should be [3, 3, 3, input_layer.get_shape()[4], filters]: double checked.
            kernel_initializer = tf.constant_initializer(conv_kernel_trilinear)
        else:
            raise ValueError('initializer=' + initializer + ' is not defined in conv3d_transpose. Valid options: "trilinear"')

    with tf.variable_scope(scope):
        net = tf.layers.conv3d_transpose(input_layer, filters, kernel_size,
                                         padding=padding,
                                         strides=strides,
                                         kernel_initializer=kernel_initializer,
                                         use_bias=use_bias,
                                         trainable=trainable)
        if bn_training is not None:
            net = tf.layers.batch_normalization(net, training=bn_training)
        if activation is not None:
            if activation == 'LReLu':
                net = tf.nn.leaky_relu(net)
            elif activation == 'ReLu':
                net = tf.nn.relu(net)
            elif activation == 'ELu':
                net = tf.nn.elu(net)
            else:
                raise ValueError('activation=' + activation + ' is not defined in tfu.conv3d. Valid options: "ReLu", "LReLu"')
    return net


def max_pooling3d(input_layer, pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid', scope=None):
    with tf.variable_scope(scope):
        net = tf.layers.max_pooling3d(input_layer, pool_size=pool_size, strides=strides, padding=padding)
    return net
