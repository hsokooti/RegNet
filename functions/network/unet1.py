import tensorflow as tf
import numpy as np
import functions.tf_utils as tfu
import logging


def network(images, bn_training, detailed_summary=False, use_keras=False):
    with tf.variable_scope('AdditiveNoise'):
        images = tf.cond(bn_training,
                         lambda: images + tf.round(tf.random_normal(tf.shape(images),
                                                                    mean=tf.round(tf.random_normal([1], mean=0, stddev=2, dtype=tf.float32)),
                                                                    stddev=2, dtype=tf.float32)),
                         lambda: images)

    common = {'padding': 'valid', 'activation': 'ReLu', 'bn_training': bn_training, 'use_keras': use_keras}
    with tf.variable_scope('R1'):
        conv1_r1 = tfu.layers.conv3d(images, 12, [3, 3, 3], scope='conv1_R1', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)

    with tf.variable_scope('R2'):
        images_r2 = tfu.layers.max_pooling3d(conv1_r1, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', scope='max_R2', use_keras=use_keras)
        conv1_r2 = tfu.layers.conv3d(images_r2, 24, [3, 3, 3], scope='conv1_R2', **common)

    with tf.variable_scope('R3'):
        images_r3 = tfu.layers.max_pooling3d(conv1_r2, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', scope='max_R3', use_keras=use_keras)
        conv1_r3 = tfu.layers.conv3d(images_r3, 28, [3, 3, 3], scope='conv1_R3', **common)

    with tf.variable_scope('R4'):
        images_r4 = tfu.layers.max_pooling3d(conv1_r3, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', scope='max_R4', use_keras=use_keras)
        conv1_r4 = tfu.layers.conv3d(images_r4, 32, [3, 3, 3], scope='conv1_R4', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)
        conv2_r4 = tfu.layers.conv3d(conv1_r4, 40, [3, 3, 3], scope='conv2_R4', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)
        conv3_r4 = tfu.layers.conv3d(conv2_r4, 44, [3, 3, 3], scope='conv3_R4', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)

    with tf.variable_scope('R3_Up'):
        conv1_r3_up = tfu.layers.upsampling3d(conv3_r4, scope='conv1_R3_Up', interpolator='trilinear')
        concat_r3_up = tf.concat([conv1_r3_up, conv1_r3], 4)
        conv2_r3 = tfu.layers.conv3d(concat_r3_up, 32, [3, 3, 3], scope='conv2_R3_Up', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)

    with tf.variable_scope('R2_Up'):
        conv1_r2_up = tfu.layers.upsampling3d(conv2_r3, scope='conv1_R2_Up', interpolator='trilinear')
        mr = 1
        concat_r2_up = tf.concat([tf.pad(conv1_r2_up, ([0, 0], [mr, mr], [mr, mr], [mr, mr], [0, 0]), constant_values=0), conv1_r2], 4)
        conv2_r2 = tfu.layers.conv3d(concat_r2_up, 18, [3, 3, 3], scope='conv2_R2_Up', padding='same', activation='ReLu', bn_training=bn_training, use_keras=use_keras)

    with tf.variable_scope('R1_Up'):
        conv1_r1_up = tfu.layers.upsampling3d(conv2_r2, scope='conv1_R1_Up', interpolator='trilinear')
        mr = 1
        concat_r1_up = tf.concat([tf.pad(conv1_r1_up, ([0, 0], [mr, mr], [mr, mr], [mr, mr], [0, 0]), constant_values=0), conv1_r1], 4)
        conv2_r1 = tfu.layers.conv3d(concat_r1_up, 12, [3, 3, 3], scope='conv1_R2_Up', padding='same', activation='ELu', bn_training=bn_training, use_keras=use_keras)

    with tf.variable_scope('DVF'):
        dvf_regnet = tfu.layers.conv3d(conv2_r1, 3, [1, 1, 1], padding="valid", activation=None, bn_training=None, scope='DVF_RegNet')

    if detailed_summary:
        for i in range(1, 8):
            tensor_name = 'conv'+str(i)+'_R1'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R1_conv'+str(i), selected_slices=1)
        for i in range(1, 9):
            tensor_name = 'conv' + str(i) + '_R2'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R2_conv'+str(i), selected_slices=1)
        for i in range(1, 10):
            tensor_name = 'conv'+str(i)+'_R4'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R4_conv'+str(i), selected_slices=1)

    return dvf_regnet


def run_network(train_mode='Train'):
    """
    :param train_mode: 'Training', 'Test'
    :return:
    """
    batch_size = 5
    d_in_val = 125

    if train_mode == 'Train':
        d_in_train, d_out_train = [2 * i + 1 for i in raidus_train()]
        d_in_tf = d_in_train
        d_out_tf = d_out_train
    elif train_mode == 'Test':
        d_in_test, d_out_test = [2 * i + 1 for i in maximum_radius_test()]
        d_in_tf = d_in_test
        d_out_tf = d_out_test
    else:
        raise ValueError('Train or Test')

    import time
    reg_weight = 0.1
    learning_rate = 1E-3
    tf.reset_default_graph()
    tf.set_random_seed(0)
    images_tf = tf.placeholder(tf.float32, shape=[None, d_in_tf, d_in_tf, d_in_tf, 2], name="Images")
    bn_training = tf.placeholder(tf.bool, name='bn_training')
    dvf_ground_truth = tf.placeholder(tf.float32, shape=[None, d_out_tf, d_out_tf, d_out_tf, 3], name="DVF_GroundTruth")
    dvf_predict = network(images_tf, bn_training, detailed_summary=False)
    huber = (tf.losses.huber_loss(dvf_ground_truth, dvf_predict, weights=1))
    with tf.variable_scope('bending_energy'):
        bending_energy = tfu.image_processing.bending_energy(dvf_predict, voxel_size=[1, 1, 1])
    loss = tf.add(huber, reg_weight * bending_energy)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
    with tf.device('/device:GPU:0'):  # Replace with device you are interested in
        bytes_in_use_tf = BytesInUse()

    if train_mode == 'Test':
        for i in range(10):
            batch_im = np.random.rand(1, d_in_test, d_in_test, d_in_test, 2)
            batch_dvf = np.random.rand(1, d_out_test, d_out_test, d_out_test, 3)
            time_before = time.time()
            byte_in_use, _ = sess.run([bytes_in_use_tf, dvf_predict],
                                      feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0})
            time_after = time.time()
            logging.info('Testing i={} is done in {:.2f}, byte_in_use={:.1f} MB'.format(i, time_after-time_before, byte_in_use/1024/1024))

    if train_mode == 'Train':
        # training
        for i in range(10):
            batch_im = np.random.rand(batch_size, d_in_train, d_in_train, d_in_train, 2)
            batch_dvf = np.random.rand(batch_size, d_out_train, d_out_train, d_out_train, 3)
            time_before = time.time()
            byte_in_use, _, _, _, _ = sess.run([bytes_in_use_tf, loss, huber, bending_energy, train_step],
                                               feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 1})
            time_after = time.time()
            logging.info('Training i={} is done in {:.2f}, byte_in_use={:.1f} MB'.format(i, time_after-time_before, byte_in_use/1024/1024))

        # validation
        for i in range(5):
            batch_im = np.random.rand(batch_size, d_in_val, d_in_val, d_in_val, 2)
            time_before = time.time()
            sess.run([dvf_predict],
                     feed_dict={images_tf: batch_im, bn_training: 0})
            time_after = time.time()
            logging.info('Validation i={} is done in {:.2f}'.format(i, time_after-time_before))


def raidus_train():
    r_input = 62
    r_output = 62
    # r_input = 66
    # r_output = 66
    return r_input, r_output


def radius_test(gpu_memory=12):
    if gpu_memory == 12:
        r_input = 62
        r_output = 62
    else:
        raise ValueError('radius for this gpu_memory is not defined')
    return r_input, r_output


def maximum_radius_test(gpu_memory=None, number_of_gpu=None):
    """
    :return:
    """
    if gpu_memory is None and number_of_gpu is None:
        gpu_memory, number_of_gpu = tfu.client.read_gpu_memory()
    logging.info('GPU Memory={:.2f} Number of GPU={}'.format(gpu_memory, number_of_gpu))
    if 10 <= gpu_memory < 11:
        r_input = 134
        r_output = 134
    elif 11 <= gpu_memory < 12:
        r_input = 62
        r_output = 26
    else:
        r_input = 62
        r_output = 26

    logging.info('crop5: GPU Memory={:.2f}, Number of GPU={}, Max radius={}'.format(gpu_memory, number_of_gpu, r_input))
    return r_input, r_output


def get_resize_unit():
    """
    Accepted value that you can add or subtract to the network radius.
    This is relevant to the number of max-pooling insided the network
    :return: resize_unit
    """
    resize_unit = 8
    return resize_unit


def get_padto():
    padto = 125
    return padto


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    d_input, d_output = [2 * i + 1 for i in raidus_train()]
    print_all = False
    do_training = True
    network(tf.placeholder(tf.float32, shape=[None, d_input, d_input, d_input, 2]),
            tf.placeholder(tf.bool, name='bn_training'))
    logging.info('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    if print_all:
        gr = tf.get_default_graph()
        for op in tf.get_default_graph().get_operations():
            logging.info(str(op.name))
    if do_training:
        run_network(train_mode='Test')
