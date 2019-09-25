import argparse
import copy
import datetime
import logging
import matplotlib as mpl
import multiprocessing
import numpy as np
import os
from pathlib import Path
import shutil
import tensorflow as tf
import time
import functions.reading as reading
import functions.general_utils as gut
import functions.setting.setting_utils as su
import functions.network as network
import functions.tf_utils as tfu


def run_regnet(setting, stage=None):
    """
    The main script to train the RegNet. It is possible to train for all stages with a for loop define in initialize()
    :param setting: an initialized setting only have the root address
    :param stage: RegNets: overwrite the stage
    :return:
    """
    load_model = False
    network_name = 'crop4_connection'  # choosing a network available in functions.network: 'decimation4', 'crop4_connection', 'unet1'
    setting['stage'] = 1               # RegNets : 4, 2, 1
    setting['AGMode'] = 'Resp'         # Artificial Generation Mode: 'Resp', 'NoResp', 'SingleOnly', 'MixedOnly', 'SingleResp' , 'Visualization'
    setting['use_keras'] = False       # Use tf.keras.layers or tf.layers
    deform_exp_s1 = '3D_max7_D14_K'    # artificial generation defined in functions.setting.experiment_setting: '3D_max7_D14_K', '3D_max15_SingleFrequency_Visualization
    setting['start_chunk'] = 0         # whether to start from the beginning or not

    if stage is not None:
        setting['stage'] = stage  # overwrite the stage if it is given

    if load_model:
        loaded_network = {'NetworkLoad': '20190304_3D_max15_D14_K_NoResp_S2_r5_dec4',
                          'GlobalStepLoad': 'Last',   # 'Auto', 'Last', '#Number'
                          'BatchSize': 'Auto',
                          'itr_load': 'Auto',
                          'semi_epoch_load': 'Auto'}
        loaded_network = su.load_network(setting, loaded_network)
        itr_load = loaded_network['itr_load']
        semi_epoch_load = loaded_network['semi_epoch_load']
    else:
        itr_load = 0
        semi_epoch_load = 0

    setting['DetailedNetworkSummary'] = False  # Write many feature maps of the network. This is an input to all networsk: functions.network.$network_name
    regularization = True                      # bending energy (BE) regularizer
    inspecting_regularization_weight = False   # a special experiment in order to find the optimal weight for adding Huber loss and BE loss
    setting['ParallelGeneration1stEpoch'] = True  # Parallel generation of images in the first epoch. Can speed up the training.
    setting['ParallelSearching'] = True           # Using np.where in parallel with [number of cores - 2] in order to make balanced data. This is done with joblib library
    setting['Randomness'] = True               # Shuffling the data. Normally should be True
    setting['PadTo'] = {'stage'+str(setting['stage']): getattr(getattr(network, network_name), 'get_padto')()}  # if the design of the network force it to pad to a fixed number

    # %%------------------------------------------------ Setting of reading DVFs ------------------------------------------------
    dsmoothlist_training, dsmoothlist_validation = su.dsmoothlist_by_deform_exp(deform_exp_s1, setting['AGMode'])
    data_exp_dict = [{'data': 'SPREAD',                               # Data to load. The image addresses can be modified in setting_utils.py
                      'deform_exp': deform_exp_s1,                    # Artificial deformation experiment
                      'TrainingCNList': [i for i in range(1, 11)],    # Case number of images to load (The patient number)
                      'TrainingTypeImList': [0, 1],                   # Types images for each case number, for example [baseline, follow-up]
                      'TrainingDSmoothList': su.repeat_dsmooth_numbers(dsmoothlist_training, deform_exp_s1, repeat=2),  # The synthetic type to load. For instance, ['single_frequency', 'respiratory_motion']
                      'TrainingDeformedImExt': ['Clean', 'Sponge', 'Noise'],  # The sequence of intensity augmentation over the deformed image: 'Clean', 'Noise', 'Occluded', 'Sponge'
                      'ValidationCNList': [11, 12],
                      'ValidationTypeImList': [0, 1],
                      'ValidationDSmoothList': dsmoothlist_validation,
                      'ValidationDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                      },
                     {'data': 'DIR-Lab_COPD',
                      'deform_exp': deform_exp_s1,
                      'TrainingCNList': [i for i in range(1, 10)],
                      'TrainingTypeImList': [0, 1],
                      'TrainingDSmoothList': su.repeat_dsmooth_numbers(dsmoothlist_training, deform_exp_s1, repeat=2),
                      'TrainingDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                      'ValidationCNList': [10],
                      'ValidationTypeImList': [0, 1],
                      'ValidationDSmoothList': dsmoothlist_validation,
                      'ValidationDeformedImExt': ['Clean', 'Sponge', 'Noise'],
                      }
                     ]

    if setting['stage'] == 1:
        data_exp_dict = copy.deepcopy(data_exp_dict)
        for i in range(len(data_exp_dict)):
            if setting['AGMode'] == 'SinlgeOnly':
                data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=5)

    elif setting['stage'] == 2:
        data_exp_dict = copy.deepcopy(data_exp_dict)
        for i in range(len(data_exp_dict)):
            data_exp_dict[i]['deform_exp'] = '3D_max15_D14_K'
            data_exp_dict[i]['ValidationDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=2)
            if setting['AGMode'] == 'Resp':
                data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=3)
            if setting['AGMode'] == 'NoResp':
                data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=5)
            if setting['AGMode'] == 'SingleOnly':
                data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=15)
    elif setting['stage'] == 4:
        data_exp_dict = copy.deepcopy(data_exp_dict)
        for i in range(len(data_exp_dict)):
            data_exp_dict[i]['deform_exp'] = '3D_max20_D14_K'
            data_exp_dict[i]['ValidationDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=5)
            data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=5)
            if setting['AGMode'] == 'SingleOnly':
                data_exp_dict[i]['TrainingDSmoothList'] = su.repeat_dsmooth_numbers(dsmoothlist_training, data_exp_dict[i]['deform_exp'], repeat=15)
    else:
        raise ValueError("setting['stage'] should be in [1, 2, 4]")
    setting = su.load_setting_from_data_dict(setting, data_exp_dict)  # load the setting of desired data
    setting = su.load_network_setting(setting, network_name=network_name)  # load the setting of the selected network like radius and pad
    setting['ClassBalanced'] = su.load_suggested_class_balanced(setting)  # ClassBlanced is used for balance sampling. You can use suggested values or
    # manually define it. eg. [2, 5, 15] --> equal number of samples in [0, 2), [2, 5), [5, 15)
    if setting['PadTo']['stage'+str(setting['stage'])] is None:
        setting['FullImage'] = False  # for some network designs (eg: unet1) we don't use patch-based training, alternatively we give the entire images as inputs
    else:
        setting['FullImage'] = True

    setting['DownSamplingByGPU'] = False
    setting['Augmentation'] = False

    # training
    setting['NetworkTraining'] = dict()
    setting['NetworkTraining']['reg_weight'] = 0.1
    setting['NetworkTraining']['MaxQueueSize'] = 20
    setting['NetworkTraining']['VoxelSize'] = [1, 1, 1]
    setting['NetworkTraining']['LearningRate_Base'] = 1E-3
    setting['NetworkTraining']['LearningRate_Decay'] = None  # 0.9
    setting['NetworkTraining']['MaxIterations'] = np.int(1E7)
    setting['NetworkValidation'] = dict()
    setting['NetworkValidation']['MaxQueueSize'] = 10

    if setting['stage'] == 1:
        setting['NetworkTraining']['NumberOfImagesPerChunk'] = 16  # Number of images that I would like to load in RAM
        setting['NetworkTraining']['SamplesPerImage'] = 50
        setting['NetworkTraining']['BatchSize'] = 15
        setting['NetworkValidation']['NumberOfImagesPerChunk'] = 10  # Number of images that I would like to load in RAM
        setting['NetworkValidation']['SamplesPerImage'] = 500
        setting['NetworkValidation']['BatchSize'] = 50

    elif setting['stage'] == 2:
        setting['NetworkTraining']['NumberOfImagesPerChunk'] = 40
        setting['NetworkTraining']['SamplesPerImage'] = 20
        setting['NetworkTraining']['BatchSize'] = 15
        setting['NetworkValidation']['NumberOfImagesPerChunk'] = 60
        setting['NetworkValidation']['SamplesPerImage'] = 10
        setting['NetworkValidation']['BatchSize'] = 50

    elif setting['stage'] == 4:
        if setting['FullImage']:
            setting['NetworkTraining']['NumberOfImagesPerChunk'] = 40
            setting['NetworkTraining']['SamplesPerImage'] = 20  # no use when using FullImage, each sample is the entire image
            setting['NetworkTraining']['BatchSize'] = 5
            setting['NetworkValidation']['NumberOfImagesPerChunk'] = 200
            setting['NetworkValidation']['SamplesPerImage'] = 50  # no use when using FullImage
            setting['NetworkValidation']['BatchSize'] = 10
        else:
            setting['NetworkTraining']['NumberOfImagesPerChunk'] = 80
            setting['NetworkTraining']['SamplesPerImage'] = 5
            setting['NetworkTraining']['BatchSize'] = 15
            setting['NetworkValidation']['NumberOfImagesPerChunk'] = 200
            setting['NetworkValidation']['SamplesPerImage'] = 10
            setting['NetworkValidation']['BatchSize'] = 50

    if setting['never_generate_image'] == 'True':
        setting['ParallelGeneration1stEpoch'] = False

    setting = backup_script(setting)
    su.check_setting(setting)
    su.write_setting(setting)

    if not semi_epoch_load:
        if setting['ParallelGeneration1stEpoch']:
            # create a new process and generate images in parallel with the main process
            generate_1st_epoch = reading.chunk_image.Images(
                setting=setting,
                class_mode='1stEpoch',
                train_mode='Training',
                number_of_images_per_chunk=setting['NetworkTraining']['NumberOfImagesPerChunk'],
                samples_per_image=setting['NetworkTraining']['SamplesPerImage'],
                im_info_list_full=su.get_im_info_list_from_train_mode(setting, train_mode='Training'),
                stage=setting['stage'],
                semi_epoch=semi_epoch_load,
                full_image=setting['FullImage'],
            )
            process = multiprocessing.Process(target=generate_1st_epoch.generate_chunk_only)
            if setting['only_generate_image']:
                process.run()
                exit()
            else:
                process.start()
    else:
        setting['ParallelGeneration1stEpoch'] = False

    reg_weight = setting['NetworkTraining']['reg_weight']
    tf.reset_default_graph()
    tf.set_random_seed(0)

    with tf.variable_scope('InputImages'):
        images_tf = tf.placeholder(tf.float32, shape=[None, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2], name="Images")
    dvf_ground_truth = tf.placeholder(tf.float32, shape=[None, 2*setting['Ry']+1, 2*setting['Ry']+1, 2*setting['Ry']+1, 3], name="DVF_GroundTruth")
    bn_training = tf.placeholder(tf.bool, name='bn_training')
    loss_average_tf = tf.placeholder(tf.float32)
    if regularization:
        huber_average_tf = tf.placeholder(tf.float32)
        bending_average_tf = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    dvf_predict = getattr(getattr(network, setting['NetworkDesign']), 'network')(images_tf, bn_training,
                                                                                 detailed_summary=setting['DetailedNetworkSummary'],
                                                                                 use_keras=setting['use_keras'])

    huber = tf.losses.huber_loss(dvf_ground_truth, dvf_predict, weights=1)
    with tf.variable_scope('bending_energy'):
        bending_energy = tfu.image_processing.bending_energy(dvf_predict, voxel_size=setting['NetworkTraining']['VoxelSize'])
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if regularization:
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
        tf.summary.image('Fixed_Images', images_tf[:, :, :, :, 0, np.newaxis][batch_summary, int(images_tf.get_shape()[1]) // 2, np.newaxis, :, :, :], 1)
        tf.summary.image('Deformed_Images', images_tf[:, :, :, :, 1, np.newaxis][batch_summary, int(images_tf.get_shape()[1]) // 2, np.newaxis, :, :, :], 1)
        tf.summary.image('GroundTruth_X', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 0, np.newaxis], 1)
        tf.summary.image('GroundTruth_Y', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 1, np.newaxis], 1)
        tf.summary.image('GroundTruth_Z', dvf_ground_truth[batch_summary, int(dvf_ground_truth.get_shape()[1]) // 2, np.newaxis, :, :, 2, np.newaxis], 1)
        tf.summary.image('RegNet_X', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 0, np.newaxis], 1)
        tf.summary.image('RegNet_Y', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 1, np.newaxis], 1)
        tf.summary.image('RegNet_Z', dvf_predict[batch_summary, int(dvf_predict.get_shape()[1]) // 2, np.newaxis, :, :, 2, np.newaxis], 1)
        tf.summary.scalar("loss", loss_average_tf)
        if regularization:
            tf.summary.scalar("Huber", huber_average_tf)
            tf.summary.scalar("BendingE", bending_average_tf)
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
        saver_loading = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver_loading.restore(sess, su.address_generator(setting, 'saved_model_with_step',
                                                         current_experiment=loaded_network['NetworkLoad'],
                                                         step=loaded_network['GlobalStepLoad']))

    train_queue = reading.train_queue.FillPatches(setting, semi_epoch=semi_epoch_load)
    train_queue.start()

    val_queue = reading.val_queue.FillPatches(setting)
    val_queue.start()

    loss_train_average = 0
    if regularization:
        huber_train_average = 0
        bending_train_average = 0
    count = 0

    for itr in range(itr_load, setting['NetworkTraining']['MaxIterations']):
        lr = setting['NetworkTraining']['LearningRate_Base']
        if setting['NetworkTraining']['LearningRate_Decay'] is not None:
            lr = setting['NetworkTraining']['LearningRate_Base'] * \
                 (setting['NetworkTraining']['LearningRate_Decay'] ** train_queue._chunk_direct._semi_epoch)

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
        logging.debug('itr={} done in CPU:{:.3f}s, GPU:{:.3f}s'.format(itr, time_after_cpu1 - time_before_cpu1, time_after_gpu - time_before_gpu))
        if train_queue.paused:   # Queue class would be paused when the queue is full
            if not train_queue._PatchQueue.full():
                train_queue.resume()
        count_itr = np.shape(batch_im)[0] / setting['NetworkTraining']['BatchSize']
        count += count_itr
        loss_train_average = loss_train_average + loss_train_itr
        if regularization:
            huber_train_average += count_itr * huber_train_itr
            bending_train_average += count_itr * bending_train_itr

        if itr % 25 == 1:
            loss_train_average = loss_train_average / count
            if regularization:
                huber_train_average = huber_train_average / count
                bending_train_average = bending_train_average / count
                [s, dvf_plot, dvf_predict_plot] = sess.run([summ, dvf_ground_truth, dvf_predict],
                                                   feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0,
                                                              loss_average_tf: loss_train_average,
                                                              huber_average_tf: huber_train_average,
                                                              bending_average_tf: bending_train_average})
                logging.info(setting['current_experiment'] + ', itr={} SemiEpoch={}, loss={:.2f} , huber={:.2f}, bendingE={:.2f}'.
                             format(itr, train_queue._chunk_direct._semi_epoch, loss_train_average, huber_train_average, bending_train_average))
                huber_train_average = 0
                bending_train_average = 0
            else:
                [s, dvf_plot, dvf_predict_plot] = sess.run([summ, dvf_ground_truth, dvf_predict],
                                                   feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0,
                                                              loss_average_tf: loss_train_average})
                logging.info('itr {} SemiEpoch = {}, loss = {:.2f}'.format(itr, train_queue._chunk_direct._semi_epoch, loss_train_average))
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
                while val_queue._PatchQueue.empty():
                    time.sleep(0.5)
                    logging.debug('waiting... Val Queue is empty :(')
                batch_im_val, batch_dvf_val, end_of_validation = val_queue._PatchQueue.get()
                time_before_gpu_val = time.time()
                count_val_itr = np.shape(batch_im_val)[0] / setting['NetworkValidation']['BatchSize']
                count_val += count_val_itr
                if regularization:
                    [loss_val_itr, huber_val_itr, bending_val_itr] = \
                        sess.run([loss, huber, bending_energy],
                                 feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0})
                    huber_val_average += count_val_itr * huber_val_itr
                    bending_val_average += count_val_itr * bending_val_itr
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
                [dvf_plot_val, dvf_predict_plot_val, s] = sess.run([dvf_ground_truth, dvf_predict, summ],
                                                           feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0,
                                                                      loss_average_tf: loss_val_average,
                                                                      huber_average_tf: huber_val_average,
                                                                      bending_average_tf: bending_val_average})
            else:
                [dvf_plot_val, dvf_predict_plot_val, s] = sess.run([dvf_ground_truth, dvf_predict, summ],
                                                           feed_dict={images_tf: batch_im_val, dvf_ground_truth: batch_dvf_val, bn_training: 0,
                                                                      loss_average_tf: loss_val_average})

            test_writer.add_summary(s, itr*setting['NetworkTraining']['BatchSize'])


def backup_script(setting):
    """
    This script does
    1. Update the current_experiment name based on stage and deform_exp name.
    2. Make a backup of the whole code
    3. Start logging.
    :param setting
    :return setting
    """
    date_now = datetime.datetime.now()
    network_name = ''
    if 'crop' in setting['NetworkDesign']:
        network_name = setting['NetworkDesign'].rsplit('_', 1)[0]
    if 'decimation' in setting['NetworkDesign']:
        network_name = 'dec' + setting['NetworkDesign'][-1]
    if 'unet' in setting['NetworkDesign']:
        network_name = 'unet' + setting['NetworkDesign'][-1]
    current_experiment = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date_now.year, date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second) +\
        '_'+setting['DataExpDict'][0]['deform_exp']+'_'+setting['AGMode']+'_S'+str(setting['stage'])+'_'+network_name

    setting['current_experiment'] = current_experiment
    if not os.path.isdir(su.address_generator(setting, 'ModelFolder')):
        os.makedirs(su.address_generator(setting, 'ModelFolder'))

    shutil.copy(Path(__file__), su.address_generator(setting, 'ModelFolder'))
    shutil.copytree(Path(__file__).parent / Path('functions'), Path(su.address_generator(setting, 'ModelFolder')) / Path('functions'))
    gut.logger.set_log_file(su.address_generator(setting, 'LogFile'))
    return setting


def initialize():
    """
    read the arguments and set the root folder
    --where_to_run: "Auto" or "Cluster" or "Root". The default value is "Auto"
                    check functions.setting.setting_utils.root_address_generator()
    --only_generate_image: "True" or "False". The default value is "False
    --never_generate_image: "True" or "False". The default value is "False"
    call the run_regnet with the initialized value for setting
    """
    mpl.rcParams['agg.path.chunksize'] = 10000
    current_experiment = ''
    parser = argparse.ArgumentParser(description='read where_to_run')
    parser.add_argument('--where_to_run', '-w',
                        help='This is an optional argument, '
                             'you choose between "Auto" or "Cluster". The default value is "Auto"')
    parser.add_argument('--only_generate_image', '-g',
                        help='This is an optional argument, '
                             'you choose between "True" or "False". The default value is "False"')
    parser.add_argument('--never_generate_image', '-ng',
                        help='This is an optional argument, '
                             'you choose between "True" or "False". The default value is "False"')
    args = parser.parse_args()
    where_to_run = args.where_to_run
    only_generate_image = args.only_generate_image
    if only_generate_image == 'True':
        only_generate_image = True
    else:
        only_generate_image = False
    never_generate_image = args.never_generate_image
    if never_generate_image == 'True':
        never_generate_image = True
    else:
        never_generate_image = False
    setting = su.initialize_setting(current_experiment=current_experiment, where_to_run=where_to_run)
    setting['only_generate_image'] = only_generate_image
    setting['never_generate_image'] = never_generate_image
    run_regnet(setting, stage=None)


if __name__ == '__main__':
    initialize()
