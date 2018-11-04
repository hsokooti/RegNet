import tensorflow as tf
import numpy as np
import functions.tf_utils as tfu


def decimation3(im, im_deformed, bn_training, detailed_summary=False):
    with tf.variable_scope('AdditiveNoise'):
        im = tf.cond(bn_training, lambda: im + tf.round(tf.random_normal(tf.shape(im), mean=0, stddev=2, dtype=tf.float32)), lambda: im)
        im_deformed = tf.cond(bn_training, lambda: im_deformed + tf.round(tf.random_normal(tf.shape(im_deformed), mean=0, stddev=2, dtype=tf.float32)), lambda: im_deformed)

    with tf.variable_scope('DownSampling_S4'):
        kernel_bspline_s4 = tfu.kernels.kernel_bspline_s4()
        im_s4 = tf.nn.convolution(im, kernel_bspline_s4, 'VALID', strides=(4, 4, 4))
        im_deformed_s4 = tf.nn.convolution(im_deformed, kernel_bspline_s4, 'VALID', strides=(4, 4, 4))
        images_s4 = tf.concat([im_s4, im_deformed_s4], axis=-1)

    with tf.variable_scope('DownSampling_S2'):
        kernel_bspline_s2 = tfu.kernels.kernel_bspline_s2()
        margin_s2 = 32
        im_s2 = tf.nn.convolution(im[:, margin_s2:-margin_s2, margin_s2:-margin_s2, margin_s2:-margin_s2, :], kernel_bspline_s2, 'VALID', strides=(2, 2, 2))
        im_deformed_s2 = tf.nn.convolution(im_deformed[:, margin_s2:-margin_s2, margin_s2:-margin_s2, margin_s2:-margin_s2, :], kernel_bspline_s2, 'VALID', strides=(2, 2, 2))
        images_s2 = tf.concat([im_s2, im_deformed_s2], axis=-1)

    margin_s1 = 49
    images_s1 = tf.concat([im[:, margin_s1:-margin_s1, margin_s1:-margin_s1, margin_s1:-margin_s1, :],
                           im_deformed[:, margin_s1:-margin_s1, margin_s1:-margin_s1, margin_s1:-margin_s1, :]],
                          axis=-1)
    common = {'padding': 'valid', 'activation': 'ReLu', 'bn_training': bn_training}

    with tf.variable_scope('R4'):
        conv1_r4 = tfu.layers.conv3d(images_s4, 16, [3, 3, 3], scope='conv1_R4', **common)
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
        conv1_r2 = tfu.layers.conv3d(images_s2, 16, [3, 3, 3], scope='conv1_R2', **common)
        conv2_r2 = tfu.layers.conv3d(conv1_r2, 20, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv2_R2', **common)
        conv3_r2 = tfu.layers.conv3d(conv2_r2, 24, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv3_R2', **common)
        conv4_r2 = tfu.layers.conv3d(conv3_r2, 30, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv4_R2', **common)
        conv5_r2 = tfu.layers.conv3d(conv4_r2, 36, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv5_R2', **common)
        conv6_r2 = tfu.layers.conv3d_transpose(conv5_r2, 40, [3, 3, 3], strides=(2, 2, 2), scope='conv6_R2',
                                               use_bias=False, initializer='trilinear', **common)
        conv7_r2 = tfu.layers.conv3d(conv6_r2, 40, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv7_R2')

    with tf.variable_scope('R1'):
        conv1_r1 = tfu.layers.conv3d(images_s1, 16, [3, 3, 3], scope='conv1_R1', **common)
        conv2_r1 = tfu.layers.conv3d(conv1_r1, 20, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv2_R1', **common)
        conv3_r1 = tfu.layers.conv3d(conv2_r1, 24, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv3_R1', **common)
        conv4_r1 = tfu.layers.conv3d(conv3_r1, 30, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv4_R1', **common)
        conv5_r1 = tfu.layers.conv3d(conv4_r1, 36, [3, 3, 3], dilation_rate=(4, 4, 4), scope='conv5_R1', **common)

    with tf.variable_scope('Merged'):
        conv_concat = tf.concat([conv5_r1, conv7_r2, conv9_r4], 4)

    with tf.variable_scope('FullyConnected'):
        conv7 = tfu.layers.conv3d(conv_concat, 120, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv1_FC')
        conv8 = tfu.layers.conv3d(conv7, 60, [3, 3, 3], padding="same", activation='ReLu', bn_training=bn_training, scope='conv2_FC')

    with tf.variable_scope('DVF'):
        dvf_regnet = tfu.layers.conv3d(conv8, 3, [1, 1, 1], padding="valid", activation=None, dilation_rate=(1, 1, 1), batch_normalization=False, scope='DVF_RegNet')

    if detailed_summary:
        print('later')
        # with tf.variable_scope('detailedSummaryImages'):
        #     tensor2summary(im_s4, 'Fixed_Images_Low4')
        #     tensor2summary(im_deformed_s4, 'Deformed_Images_Low4')
        #     tensor2summary(im_s2, 'Fixed_Images_Low2')
        #     tensor2summary(im_deformed_s2, 'Deformed_Images_Low2')
        #     tensor2summary(conv1_R4, 'conv1_R4')
        #     tensor2summary(conv2_R4, 'conv2_R4')
        #     tensor2summary(conv3_R4, 'conv3_R4')
        #     tensor2summary(conv4_R4, 'conv4_R4')
        #     tensor2summary(conv5_R4, 'conv5_R4')
        #     tensor2summary(conv6_R4, 'conv6_R4')
        #     tensor2summary(conv7_R4, 'conv7_R4')
        #     tensor2summary(conv8_R4, 'conv8_R4')
        #     tensor2summary(conv9_R4, 'conv9_R4')
        #
        #     tensor2summary(conv1_R2, 'conv1_R2')
        #     tensor2summary(conv2_R2, 'conv2_R2')
        #     tensor2summary(conv3_R2, 'conv3_R2')
        #     tensor2summary(conv4_R2, 'conv4_R2')
        #     tensor2summary(conv5_R2, 'conv5_R2')
        #     tensor2summary(conv6_R2, 'conv6_R2')
        #     tensor2summary(conv7_R2, 'conv7_R2')
        #
        #     tensor2summary(conv1_R1, 'conv1_R1')
        #     tensor2summary(conv2_R1, 'conv2_R1')
        #     tensor2summary(conv3_R1, 'conv3_R1')
        #     tensor2summary(conv4_R1, 'conv4_R1')
        #     tensor2summary(conv5_R1, 'conv5_R1')
        #
        #     tensor2summary(conv7, 'conv7')
        #     tensor2summary(conv8, 'conv8')

    return dvf_regnet


if __name__ == '__main__':
    r_input = 155
    decimation3(tf.placeholder(tf.float32, shape=[None, r_input, r_input, r_input, 1]),
                tf.placeholder(tf.float32, shape=[None, r_input, r_input, r_input, 1]),
                tf.placeholder(tf.bool, name='bn_training'), True)
    print('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
