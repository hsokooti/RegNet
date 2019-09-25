import tensorflow as tf
import numpy as np
import functions.tf_utils as tfu


def network(im, im_deformed, bn_training, detailed_summary=False):
    common = {'padding': 'valid', 'activation': 'ReLu', 'bn_training': bn_training}

    with tf.variable_scope('AdditiveNoise'):
        im = tf.cond(bn_training, lambda: im + tf.round(tf.random_normal(tf.shape(im), mean=0, stddev=2, dtype=tf.float32)), lambda: im)
        im_deformed = tf.cond(bn_training, lambda: im_deformed + tf.round(tf.random_normal(tf.shape(im_deformed), mean=0, stddev=2, dtype=tf.float32)), lambda: im_deformed)

    with tf.variable_scope('DownSampling_R4'):
        kernel_bspline_r4 = tfu.kernels.kernel_bspline_r4()
        im_r4 = tf.nn.convolution(im, kernel_bspline_r4, 'VALID', strides=(4, 4, 4))
        im_deformed_r4 = tf.nn.convolution(im_deformed, kernel_bspline_r4, 'VALID', strides=(4, 4, 4))
        images_r4 = tf.concat([im_r4, im_deformed_r4], axis=-1)

    with tf.variable_scope('DownSampling_R2'):
        kernel_bspline_r2 = tfu.kernels.kernel_bspline_r2()
        margin_r2 = 32
        im_r2 = tf.nn.convolution(im[:, margin_r2:-margin_r2, margin_r2:-margin_r2, margin_r2:-margin_r2, :], kernel_bspline_r2, 'VALID', strides=(2, 2, 2))
        im_deformed_r2 = tf.nn.convolution(im_deformed[:, margin_r2:-margin_r2, margin_r2:-margin_r2, margin_r2:-margin_r2, :], kernel_bspline_r2, 'VALID', strides=(2, 2, 2))
        images_r2 = tf.concat([im_r2, im_deformed_r2], axis=-1)

    with tf.variable_scope('DownSampling_R1'):
        margin_r1 = 49
        images_r1 = tf.concat([im[:, margin_r1:-margin_r1, margin_r1:-margin_r1, margin_r1:-margin_r1, :],
                               im_deformed[:, margin_r1:-margin_r1, margin_r1:-margin_r1, margin_r1:-margin_r1, :]],
                              axis=-1)

    with tf.variable_scope('R4'):
        conv1_r4 = tfu.layers.conv3d(images_r4, 16, [3, 3, 3], scope='conv1_R4', **common)
        conv2_r4 = tfu.layers.conv3d(conv1_r4, 20, [3, 3, 3],  dilation_rate=(2, 2, 2), scope='conv2_R4', **common)
        conv3_r4 = tfu.layers.conv3d(conv2_r4, 24, [3, 3, 3],  dilation_rate=(4, 4, 4), scope='conv3_R4', **common)
        conv4_r4 = tfu.layers.conv3d(conv3_r4, 30, [3, 3, 3],  dilation_rate=(4, 4, 4), scope='conv4_R4', **common)
        conv5_r4 = tfu.layers.conv3d(conv4_r4, 36, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv5_R4', **common)
        conv6_r4 = tfu.layers.conv3d_transpose(conv5_r4, 40, [3, 3, 3], strides=(2, 2, 2), scope='conv6_R4',
                                               use_bias=False, initializer='trilinear', **common)
        conv7_r4 = tfu.layers.conv3d(conv6_r4, 45, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv7_R4')
        conv8_r4 = tfu.layers.conv3d_transpose(conv7_r4, 50, [3, 3, 3], padding="valid", activation='ReLu', strides=(2, 2, 2),
                                               bn_training=bn_training, scope='conv8_R4', use_bias=False, initializer='trilinear')
        conv9_r4 = tfu.layers.conv3d(conv8_r4, 50, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv9_R4')

    with tf.variable_scope('R2'):
        conv1_r2 = tfu.layers.conv3d(images_r2, 16, [3, 3, 3], scope='conv1_R2', **common)
        conv2_r2 = tfu.layers.conv3d(conv1_r2, 20, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv2_R2', **common)
        conv3_r2 = tfu.layers.conv3d(conv2_r2, 24, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv3_R2', **common)
        conv4_r2 = tfu.layers.conv3d(conv3_r2, 30, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv4_R2', **common)
        conv5_r2 = tfu.layers.conv3d(conv4_r2, 36, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv5_R2', **common)
        conv6_r2 = tfu.layers.conv3d_transpose(conv5_r2, 40, [3, 3, 3], strides=(2, 2, 2), scope='conv6_R2',
                                               use_bias=False, initializer='trilinear', **common)
        conv7_r2 = tfu.layers.conv3d(conv6_r2, 40, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv7_R2')

    with tf.variable_scope('R1'):
        conv1_r1 = tfu.layers.conv3d(images_r1, 16, [3, 3, 3], scope='conv1_R1', **common)
        conv2_r1 = tfu.layers.conv3d(conv1_r1, 20, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv2_R1', **common)
        conv3_r1 = tfu.layers.conv3d(conv2_r1, 24, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv3_R1', **common)
        conv4_r1 = tfu.layers.conv3d(conv3_r1, 30, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv4_R1', **common)
        conv5_r1 = tfu.layers.conv3d(conv4_r1, 36, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv5_R1', **common)

    with tf.variable_scope('Merged'):
        conv_concat = tf.concat([conv5_r1, conv7_r2, conv9_r4], 4)

    with tf.variable_scope('FullyConnected'):
        conv1_fc = tfu.layers.conv3d(conv_concat, 120, [3, 3, 3], padding="same", activation='ELu', bn_training=bn_training, scope='conv1_FC')
        conv2_fc = tfu.layers.conv3d(conv1_fc, 60, [3, 3, 3], padding="same", activation='ELu', bn_training=bn_training, scope='conv2_FC')

    with tf.variable_scope('DVF'):
        dvf_regnet = tfu.layers.conv3d(conv2_fc, 3, [1, 1, 1], padding="valid", activation=None, bn_training=None, scope='DVF_RegNet')

    if detailed_summary:
        for i in range(1, 6):
            tensor_name = 'conv'+str(i)+'_R1'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R1_conv'+str(i), selected_slices=1)
        for i in range(1, 8):
            tensor_name = 'conv' + str(i) + '_R2'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R2_conv'+str(i), selected_slices=1)
        for i in range(1, 10):
            tensor_name = 'conv'+str(i)+'_R4'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R4_conv'+str(i), selected_slices=1)
        tfu.summary.tensor2summary(conv1_fc, 'conv1_FC', scope='DetailedSummaryImages_conv1_FC', selected_slices=1)
        tfu.summary.tensor2summary(conv2_fc, 'conv2_FC', scope='DetailedSummaryImages_conv2_FC', selected_slices=1)

    return dvf_regnet


def raidus_train():
    r_input = 77
    r_output = 13
    return r_input, r_output


def radius_test():
    r_input = 127
    r_output = 63
    return r_input, r_output


if __name__ == '__main__':
    d_input, d_output = [2 * i + 1 for i in raidus_train()]
    network(tf.placeholder(tf.float32, shape=[None, d_input, d_input, d_input, 1]),
            tf.placeholder(tf.float32, shape=[None, d_input, d_input, d_input, 1]),
            tf.placeholder(tf.bool, name='bn_training'), True)
    print('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
