import argparse
import datetime
import logging
import matplotlib as mpl
import multiprocessing
import numpy as np
import os
import shutil
import tensorflow as tf
import time
import functions.reading as reading
# import functions.plots as plots
import functions.general_utils as gut
import functions.setting.setting_utils as su
import functions.artificial_generation.dvf_generation as synth
import functions.tf_utils as tfu


def run_regnet(setting, base_learning_rate=1E-4, max_steps=1000):

    # %%------------------------------------------- Setting of the network ------------------------------------------
    load_model = False
    network_name = 'decimation3'  # 'decimation3', 'crop4_connection'
    setting['stage'] = 1

    if load_model:
        exp_load = '20180921_max15_D12_stage2_crop3'
        global_step_load = su.load_global_step_from_current_experiment(exp_load)
        batch_size_load = 15
        itr_load = int(int(global_step_load) / batch_size_load) + 1
        semi_epoch_load = 373
    else:
        itr_load = 0
        semi_epoch_load = 0

    setting['DetailedSummary'] = False
    regularization = True
    inspecting_regularization_weight = False
    # %%------------------------------------------------ Setting of reading DVFs ------------------------------------------------
    if setting['stage'] == 1:
        data_exp_dict = [{'data': 'SPREAD',
                          'deform_exp': '3D_max7_D9',
                          'TrainingCNList': [i for i in range(1, 11)],
                          'TrainingTypeImList': [0, 1],
                          'TrainingDSmoothList': [i for i in range(9)],
                          'ValidationCNList': [11, 12],
                          'ValidationTypeImList': [0, 1],
                          'ValidationDSmoothList': [2, 4, 8],
                          },
                         {'data': 'DIR-Lab_4D',
                          'deform_exp': '3D_max7_D9',
                          'TrainingCNList': [1, 2, 3],
                          'TrainingTypeImList': [i for i in range(8)],
                          'TrainingDSmoothList': [i for i in range(9)],
                          'ValidationCNList': [1, 2],
                          'ValidationTypeImList': [8, 9],
                          'ValidationDSmoothList': [2, 4, 8],
                          }
                         ]
    elif setting['stage'] == 2:
        data_exp_dict = [{'data': 'SPREAD', 'deform_exp': '3D_max15_D12',
                          'TrainingDSmoothList': [i for i in range(12)],
                          'TrainingCNList': [i for i in range(1, 11)],
                          'TrainingTypeImList': [0, 1],
                          'ValidationDSmoothList': [1, 5, 10],
                          'ValidationCNList': [11, 12],
                          'ValidationTypeImList': [0, 1]},

                         {'data': 'DIR-Lab_4D', 'deform_exp': '3D_max15_D12',
                          'TrainingDSmoothList': [i for i in range(12)],
                          'TrainingCNList': [1, 2, 3],
                          'TrainingTypeImList': [i for i in range(8)],
                          'ValidationDSmoothList': [1, 5, 10],
                          'ValidationCNList': [1, 2],
                          'ValidationTypeImList': [8, 9]},
                         ]
    elif setting['stage'] == 4:
        data_exp_dict = [{'data': 'SPREAD', 'deform_exp': '3D_max20_D12',
                          'TrainingDSmoothList': [i for i in range(12)],
                          'TrainingCNList': [i for i in range(1, 11)],
                          'TrainingTypeImList': [0, 1],
                          'ValidationDSmoothList': [1, 5, 10],
                          'ValidationCNList': [11, 12],
                          'ValidationTypeImList': [0, 1]},

                         {'data': 'DIR-Lab_4D', 'deform_exp': '3D_max20_D12',
                          'TrainingDSmoothList': [i for i in range(12)],
                          'TrainingCNList': [1, 2, 3],
                          'TrainingTypeImList': [i for i in range(8)],
                          'ValidationDSmoothList': [1, 5, 10],
                          'ValidationCNList': [1, 2],
                          'ValidationTypeImList': [8, 9]},
                         ]
    else:
        raise ValueError("setting['stage'] should be in [1, 2, 4]")
    setting = su.load_setting_from_data_dict(setting, data_exp_dict)
    setting = su.load_network_setting(setting, network_name=network_name)
    # setting['classBalanced'] = [1.5, 4, 10]           # Use these threshold values to balance the number of data in each category. for instance [a,b] implies classes [0,a), [a,b). Numbers are in mm
    # setting['classBalanced'] = [2, 7]
    # setting['classBalanced'] = [1.5, 8, 20]
    setting['classBalanced'] = su.load_class_balanced(setting)
    setting['R'] = setting['NetworkInputSize']                                   # Radius of normal resolution patch size. Total size is (2*R +1)
    setting['Ry'] = setting['NetworkOutputSize']                                  # Radius of output. Total size is (2*Ry +1)
    setting['ImPad_S1'] = setting['NetworkInputSize']                            # Pad images with setting['defaultPixelValue']
    setting['ImPad_S2'] = setting['NetworkInputSize']
    setting['ImPad_S4'] = setting['NetworkInputSize']
    setting['DVFPad_S1'] = 0
    setting['DVFPad_S2'] = 0
    setting['DVFPad_S4'] = 0
    setting['ParallelImageGeneration'] = True
    setting['ParallelProcessing'] = True                # Using np.where in parallel with [number of cores - 2] in order to make balanced data. This is done with joblib library
    setting['normalization'] = None                     # The method to normalize the intensities: 'linear'
    setting['verbose'] = True                           # Detailed printing
    setting['Dim'] = '3D'
    setting['StagesToGenerateSimultaneously'] = [2, 4]
    setting['Margin'] = setting['Ry'] + 1                               # Margin from the border to select random patches, in DVF numpy array, not Im numpy array
    setting['Randomness'] = True
    setting['Augmentation'] = False

    # training
    setting['NetworkTraining'] = dict()
    if setting['stage'] in [1, 2]:
        setting['NetworkTraining']['reg_weight'] = 0.1
        setting['NetworkTraining']['NumberOfImagesPerChunk'] = 16  # Number of images that I would like to load in RAM
        setting['NetworkTraining']['SamplesPerImage'] = 50
        setting['NetworkTraining']['BatchSize'] = 15
        setting['NetworkTraining']['MaxQueueSize'] = 20

        setting['NetworkValidation'] = dict()
        setting['NetworkValidation']['NumberOfImagesPerChunk'] = 10  # Number of images that I would like to load in RAM
        setting['NetworkValidation']['SamplesPerImage'] = 500
        setting['NetworkValidation']['BatchSize'] = 50
        setting['NetworkValidation']['MaxQueueSize'] = 10

    elif setting['stage'] == 4:
        setting['NetworkTraining']['reg_weight'] = 0.1
        setting['NetworkTraining']['NumberOfImagesPerChunk'] = 42  # Number of images that I would like to load in RAM
        setting['NetworkTraining']['SamplesPerImage'] = 20
        setting['NetworkTraining']['BatchSize'] = 15
        setting['NetworkTraining']['MaxQueueSize'] = 20

        setting['NetworkValidation'] = dict()
        setting['NetworkValidation']['NumberOfImagesPerChunk'] = 20  # Number of images that I would like to load in RAM
        setting['NetworkValidation']['SamplesPerImage'] = 50
        setting['NetworkValidation']['BatchSize'] = 50
        setting['NetworkValidation']['MaxQueueSize'] = 10

    if setting['where_to_run'] == 'Cluster':
        # Now willing to occupy multiple core in cluster mode
        setting['ParallelImageGeneration'] = False
        setting['ParallelProcessing'] = False
    su.check_setting(setting)
    su.write_setting(setting)

    if not semi_epoch_load:
        if setting['stage'] > 1:
            # We have to create all downsampled images before training the network. Because downsampling is done by GPU and we assume that one GPU is available.
            # All downsampled images are created in one loop which is much more efficient. For instance, stage 2 and 4 are created at the same time.

            im_list_info_training = su.get_im_info_list_from_train_mode(setting, train_mode='Training')
            im_list_info_validation_full = su.get_im_info_list_from_train_mode(setting, train_mode='Validation')
            im_list_info_validation = reading.utils.select_im_from_semiepoch(setting,
                                                                             im_info_list_full=im_list_info_validation_full,
                                                                             semi_epoch=0,
                                                                             chunk=0,
                                                                             number_of_images_per_chunk=setting['NetworkValidation']['NumberOfImagesPerChunk'])
            im_list_info_merged = im_list_info_training + im_list_info_validation
            for im_info in im_list_info_merged:
                synth.get_dvf_and_deformed_images(setting,
                                                  im_info=im_info,
                                                  stage=setting['stage'],
                                                  stages_to_generate_simultaneously=setting['StagesToGenerateSimultaneously'],
                                                  generation_only=True)
        generate_1st_epoch = reading.direct_1st_epoch.Images(setting=setting,
                                                             train_mode='Training',
                                                             number_of_images_per_chunk=setting['NetworkTraining']['NumberOfImagesPerChunk'],
                                                             samples_per_image=setting['NetworkTraining']['SamplesPerImage'],
                                                             im_info_list_full=su.get_im_info_list_from_train_mode(setting, train_mode='Training'),
                                                             stage=setting['stage'],
                                                             semi_epoch=semi_epoch_load)
        if setting['ParallelImageGeneration']:
            process = multiprocessing.Process(target=generate_1st_epoch.run)  # there should be no parentheses after the function!
            process.start()  # If I start this process after making my tensorflow graph, tensorboard does not work correctly!
        else:
            generate_1st_epoch.run()

    reg_weight = setting['NetworkTraining']['reg_weight']
    tf.reset_default_graph()
    with tf.variable_scope('InputImages'):
        images_tf = tf.placeholder(tf.float32, shape=[None, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2], name="Images")
        x_fixed = images_tf[:, :, :, :, 0, np.newaxis]
        x_deformed = images_tf[:, :, :, :, 1, np.newaxis]
    dvf_ground_truth = tf.placeholder(tf.float32, shape=[None, 2*setting['Ry']+1, 2*setting['Ry']+1, 2*setting['Ry']+1, 3], name="DVF_GroundTruth")
    bn_training = tf.placeholder(tf.bool, name='bn_training')
    loss_average_tf = tf.placeholder(tf.float32)
    if regularization:
        huber_average_tf = tf.placeholder(tf.float32)
        bending_average_tf = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    dvf_predict = getattr(network, setting['NetworkDesign'])(x_fixed, x_deformed, bn_training, detailed_summary=setting['DetailedSummary'])

    huber = (tf.losses.huber_loss(dvf_ground_truth, dvf_predict, weights=1))
    with tf.variable_scope('bending_energy'):
        bending_energy = tfu.image_processing.bending_energy(dvf_predict, voxel_size=setting['voxelSize'])
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if regularization:
        # loss = tf.add(huber, regWeight * bending_energy)
        if inspecting_regularization_weight:
            with tf.control_dependencies(extra_update_ops):
                train_step, mag_grad_loss, mag_grad_regularizer, regularization_weight = tf.train.AdamOptimizer(learning_rate).minimize_and_gradient(huber, bending_energy)
                loss = huber + regularization_weight * bending_energy
        else:
            loss = tf.add(huber, reg_weight * bending_energy)
            with tf.control_dependencies(extra_update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        loss = huber
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.variable_scope('Summary'):
        batch_summary = tf.shape(images_tf)[0] // 2
        tf.summary.image('Fixed_Images', x_fixed[batch_summary, int(images_tf.get_shape()[1]) // 2, np.newaxis, :, :, :], 1)
        tf.summary.image('Deformed_Images', x_deformed[batch_summary, int(images_tf.get_shape()[1]) // 2, np.newaxis, :, :, :], 1)
        tf.summary.image('GroundTruth_X', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 0, np.newaxis], 1)
        tf.summary.image('GroundTruth_Y', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 1, np.newaxis], 1)
        tf.summary.image('GroundTruth_Z', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 2, np.newaxis], 1)
        tf.summary.image('RegNet_X', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 0, np.newaxis], 1)
        tf.summary.image('RegNet_Y', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 1, np.newaxis], 1)
        tf.summary.image('RegNet_Z', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 2, np.newaxis], 1)
        tf.summary.scalar("loss", loss_average_tf)
    if regularization:
        tf.summary.scalar("huber", huber_average_tf)
        tf.summary.scalar("bendingE", bending_average_tf)
        if inspecting_regularization_weight:
            tf.summary.scalar("mag_grad_loss", mag_grad_loss)
            tf.summary.scalar("mag_grad_regularizer", mag_grad_regularizer)
            tf.summary.scalar("regularization_weight", regularization_weight)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=25, keep_checkpoint_every_n_hours=1)
    summ = tf.summary.merge_all()
    logging.debug('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(su.address_generator(setting, 'summary_train'), sess.graph)
    test_writer = tf.summary.FileWriter(su.address_generator(setting, 'summary_test'), sess.graph)
    sess.run(tf.global_variables_initializer())

    if load_model:
        all_scopes = ['R4', 'R2', 'R1', 'FullyConnected', 'DVF']
        for scope in all_scopes:
            saver_loading = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
            saver_loading.restore(sess, su.address_generator(setting, 'saved_model_with_step',
                                                             current_experiment=exp_load,
                                                             step=global_step_load))

    train_queue = reading.train_queue.FillPatches(setting, semi_epoch=semi_epoch_load)
    train_queue.start()

    val_queue = reading.val_queue.FillPatches(setting)
    val_queue.start()

    loss_train_average = 0
    if regularization:
        huber_train_average = 0
        bending_train_average = 0
    count = 0

    for itr in range(itr_load, max_steps):
        # lr_decay = 0.9
        # lr = base_learning_rate * (lr_decay ** train_queue._reading._semi_epoch)
        lr = base_learning_rate
        time_before_cpu1 = time.time()
        while train_queue._PatchQueue.empty():
            time.sleep(0.5)
            logging.debug('waiting... Train Queue is empty :(')

        batch_im, batch_dvf = train_queue._PatchQueue.get()
        time_after_cpu1 = time.time()
        time_before_gpu = time.time()

        if regularization:
            [loss_train_itr, huber_train_itr, bending_train_itr, _] = \
                sess.run([loss, huber, bending_energy, train_step],
                         feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 1, learning_rate: lr})
        else:
            [loss_train_itr, _] = sess.run([loss, train_step],
                                           feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf,  bn_training: 1, learning_rate: lr})
        time_after_gpu = time.time()
        logging.debug('itr = {} done in CPU:{:.3f}s, GPU:{:.3f}s'.format(itr, time_after_cpu1 - time_before_cpu1, time_after_gpu - time_before_gpu))
        if train_queue.paused:   # Queue class would be paused when the queue is full
            if not train_queue._PatchQueue.full():
                train_queue.resume()
        loss_train_average = loss_train_average + loss_train_itr
        if regularization:
            huber_train_average = huber_train_average + huber_train_itr
            bending_train_average = bending_train_average + bending_train_itr
        count = count + 1

        if itr % 25 == 1:
            loss_train_average = loss_train_average / count
            if regularization:
                huber_train_average = huber_train_average / count
                bending_train_average = bending_train_average / count
                [s, y_plot, y_hat_plot] = sess.run([summ, dvf_ground_truth, dvf_predict],
                                                   feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0,
                                                              loss_average_tf: loss_train_average,
                                                              huber_average_tf: huber_train_average,
                                                              bending_average_tf: bending_train_average})
                logging.info(setting['current_experiment'] + 'itr {} semiEpoch = {}, loss = {:.2f} , huber = {:.2f}, bendingE = {:.2f}'.
                             format(itr, train_queue._reading._semi_epoch, loss_train_average, huber_train_average, bending_train_average))
                huber_train_average = 0
                bending_train_average = 0
            else:
                [s, y_plot, y_hat_plot] = sess.run([summ, dvf_ground_truth, dvf_predict],
                                                   feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0,
                                                              loss_average_tf: loss_train_average})
                logging.info('itr {} semiEpoch = {}, loss = {:.2f}'.format(itr, train_queue._reading._semi_epoch, loss_train_average))
            train_writer.add_summary(s, itr * setting['NetworkTraining']['BatchSize'])
            loss_train_average = 0
            count = 0

        if itr % 1000 == 1:
            saver.save(sess, su.address_generator(setting, 'saved_model'), global_step=itr * setting['NetworkTraining']['BatchSize'])
            loss_val_average = 0
            count_val = 0
            if regularization:
                huber_val_average = 0
                bending_val_average = 0
            end_of_validation = False
            while not end_of_validation:
                count_val = count_val + 1
                while val_queue._PatchQueue.empty():
                    time.sleep(0.5)
                    logging.debug('waiting... Val Queue is empty :(')
                batch_im_val, batch_dvf_val, end_of_validation = val_queue._PatchQueue.get()
                time_before_gpu_val = time.time()
                if regularization:
                    [loss_val_itr, huber_val_itr, bending_val_itr] = \
                        sess.run([loss, huber, bending_energy],
                                 feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0})
                    huber_val_average = huber_val_average + huber_val_itr
                    bending_val_average = bending_val_average + bending_val_itr
                else:
                    [loss_val_itr] = sess.run([loss], feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0})
                time_after_gpu_val = time.time()
                logging.debug('itrVal = {} done in {:.3f}s GPU validation'.format(count_val, time_after_gpu_val - time_before_gpu_val))
                loss_val_average = loss_val_average + loss_val_itr
                if val_queue.paused:  # Queue class would be paused when the queue is full
                    if not val_queue._PatchQueue.full():
                        val_queue.resume()
            val_queue._end_of_validation = False
            loss_val_average = loss_val_average / count_val
            if regularization:
                huber_val_average = huber_val_average / count_val
                bending_val_average = bending_val_average / count_val
                [y_plot_val, y_hat_plot_val, s] = sess.run([dvf_ground_truth, dvf_predict, summ],
                                                           feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0,
                                                                      loss_average_tf: loss_val_average,
                                                                      huber_average_tf: huber_val_average,
                                                                      bending_average_tf: bending_val_average})
            else:
                [y_plot_val, y_hat_plot_val, s] = sess.run([dvf_ground_truth, dvf_predict, summ],
                                                           feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0,
                                                                      loss_average_tf: loss_val_average})

            test_writer.add_summary(s, itr*setting['NetworkTraining']['BatchSize'])


def initialize():
    base_learning_rate = 1E-3
    max_steps = np.int(1E7)
    mpl.rcParams['agg.path.chunksize'] = 10000
    date_now = datetime.datetime.now()
    current_experiment = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date_now.year, date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)
    parser = argparse.ArgumentParser(description='read where_to_run')
    parser.add_argument('--where_to_run', '-w', help='This is an optional argument, you choose between "Auto" or "Cluster". The default value is "Auto"')
    args = parser.parse_args()
    where_to_run = args.where_to_run
    setting = su.initialize_setting(current_experiment=current_experiment, where_to_run=where_to_run)
    if not os.path.isdir(su.address_generator(setting, 'Model_folder')):
        os.makedirs(su.address_generator(setting, 'Model_folder'))
    shutil.copy(os.path.realpath(__file__), su.address_generator(setting, 'Model_folder'))
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)) + '/functions/', su.address_generator(setting, 'Model_folder') + 'functions/')
    gut.logger.set_log_file(su.address_generator(setting, 'log_file'))
    run_regnet(setting, base_learning_rate=base_learning_rate, max_steps=max_steps)


if __name__ == '__main__':
    initialize()
