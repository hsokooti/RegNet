import logging
import numpy as np
import os
import SimpleITK as sitk
import time
import tensorflow as tf
import functions.reading.real_pair as real_pair
import functions.RegNetModel as RegNetModel
import functions.setting_utils as su
import functions.image_processing as ip


def multi_stage(setting, network_dict, pair_info, overwrite=False):
    """
    :param setting:
    :param network_dict:
    :param pair_info: information of the pair to be registered.
    :param overwrite:
    :return: The output moved images and dvf will be written to the disk.
             1: registration is performed correctly
             2: skip overwriting
    """
    stage_list = setting['ImagePyramidSchedule']
    final_moved_image_address = su.address_generator(setting, 'moved_image', pair_info=pair_info, stage=0, stage_list=stage_list)
    if os.path.isfile(final_moved_image_address):
        if not overwrite:
            print('overwrite=False, file '+final_moved_image_address+' already exists, skipping .....')
            return 2
        else:
            print('overwrite=True, file '+final_moved_image_address+' already exists, but overwriting .....')

    pair_stage1 = real_pair.Images(setting, pair_info, stage=1)
    pyr = dict()  # pyr: a dictionary of pyramid images
    pyr['fixed_im_s1_sitk'] = pair_stage1.get_fixed_im_sitk()
    pyr['moving_im_s1_sitk'] = pair_stage1.get_moved_im_affine_sitk()
    pyr['fixed_im_s1'] = pair_stage1.get_fixed_im()
    pyr['moving_im_s1'] = pair_stage1.get_moved_im_affine()
    if setting['torsoMask']:
        pyr['fixed_torso_s1_sitk'] = pair_stage1.get_fixed_torso_sitk()
        pyr['moving_torso_s1_sitk'] = pair_stage1.get_moved_torso_affine_sitk()
    if not (os.path.isdir(su.address_generator(setting, 'full_reg_folder', pair_info=pair_info, stage_list=stage_list))):
        os.makedirs(su.address_generator(setting, 'full_reg_folder', pair_info=pair_info, stage_list=stage_list))

    time_before_dvf = time.time()
    for i_stage, stage in enumerate(setting['ImagePyramidSchedule']):
        if stage != 1:
            pyr['fixed_im_s'+str(stage)+'_sitk'] = ip.downsampler_gpu(pyr['fixed_im_s1_sitk'], stage,
                                                                      default_pixel_value=setting['data'][pair_info[0]['data']]['defaultPixelValue'])
            pyr['moving_im_s'+str(stage)+'_sitk'] = ip.downsampler_gpu(pyr['moving_im_s1_sitk'], stage,
                                                                       default_pixel_value=setting['data'][pair_info[1]['data']]['defaultPixelValue'])
        if setting['torsoMask']:
            pyr['fixed_torso_s'+str(stage)+'_sitk'] = ip.downsampler_sitk(pyr['fixed_torso_s1_sitk'],
                                                                          stage,
                                                                          im_ref=pyr['fixed_im_s' + str(stage) + '_sitk'],
                                                                          default_pixel_value=0,
                                                                          interpolator=sitk.sitkNearestNeighbor)
            pyr['moving_torso_s'+str(stage)+'_sitk'] = ip.downsampler_sitk(pyr['moving_torso_s1_sitk'],
                                                                           stage,
                                                                           im_ref=pyr['moving_im_s' + str(stage) + '_sitk'],
                                                                           default_pixel_value=0,
                                                                           interpolator=sitk.sitkNearestNeighbor)
        else:
            pyr['fixed_torso_s'+str(stage)+'_sitk'] = None
            pyr['moving_torso_s'+str(stage)+'_sitk'] = None
        input_regnet_moving_torso = None
        if i_stage == 0:
            input_regnet_moving = 'moving_im_s'+str(stage)+'_sitk'
            if setting['torsoMask']:
                input_regnet_moving_torso = 'moving_torso_s'+str(stage)+'_sitk'
        else:
            previous_pyramid = setting['ImagePyramidSchedule'][i_stage - 1]
            dvf_composed_previous_up_sitk = 'DVF_s'+str(previous_pyramid)+'_composed_up_sitk'
            dvf_composed_previous_sitk = 'DVF_s'+str(previous_pyramid)+'_composed_sitk'
            if i_stage == 1:
                pyr[dvf_composed_previous_sitk] = pyr['DVF_s'+str(setting['ImagePyramidSchedule'][i_stage-1])+'_sitk']
            elif i_stage > 1:
                pyr[dvf_composed_previous_sitk] = sitk.Add(pyr['DVF_s'+str(setting['ImagePyramidSchedule'][i_stage-2])+'_composed_up_sitk'],
                                                           pyr['DVF_s'+str(setting['ImagePyramidSchedule'][i_stage - 1])+'_sitk'])
            pyr[dvf_composed_previous_up_sitk] = ip.upsampler_gpu(pyr[dvf_composed_previous_sitk],
                                                                  round(previous_pyramid/stage),
                                                                  dvf_output_size=pyr['fixed_im_s'+str(stage)+'_sitk'].GetSize()[::-1],
                                                                  )
            if setting['WriteAfterEachStage']:
                sitk.WriteImage(sitk.Cast(pyr[dvf_composed_previous_up_sitk], sitk.sitkVectorFloat32),
                                su.address_generator(setting, 'dvf_s_up', pair_info=pair_info, stage=previous_pyramid, stage_list=stage_list))

            dvf_t = sitk.DisplacementFieldTransform(pyr[dvf_composed_previous_up_sitk])
            # after this line DVF_composed_previous_up_sitk is converted to a transform. so we need to load it again.
            pyr['moved_im_s'+str(stage)+'_sitk'] = ip.resampler_by_dvf(pyr['moving_im_s' + str(stage)+'_sitk'],
                                                                       dvf_t,
                                                                       default_pixel_value=setting['data'][pair_info[1]['data']]['defaultPixelValue'])
            if setting['torsoMask']:
                pyr['moved_torso_s'+str(stage)+'_sitk'] = ip.resampler_by_dvf(pyr['moving_torso_s'+str(stage)+'_sitk'],
                                                                              dvf_t,
                                                                              default_pixel_value=0,
                                                                              interpolator=sitk.sitkNearestNeighbor)
            pyr[dvf_composed_previous_up_sitk] = dvf_t.GetDisplacementField()
            if setting['WriteAfterEachStage']:
                sitk.WriteImage(sitk.Cast(pyr['moved_im_s'+str(stage)+'_sitk'], setting['data'][pair_info[1]['data']]['imageByte']),
                                su.address_generator(setting, 'moved_image', pair_info=pair_info, stage=stage, stage_list=stage_list))
            input_regnet_moving = 'moved_im_s'+str(stage)+'_sitk'
            if setting['torsoMask']:
                input_regnet_moving_torso = 'moved_torso_s'+str(stage)+'_sitk'

        pyr['DVF_s'+str(stage)] = np.zeros(np.r_[pyr['fixed_im_s'+str(stage)+'_sitk'].GetSize()[::-1], 3], dtype=np.float64)
        pair_pyramid = real_pair.Images(setting, pair_info, stage=stage,
                                        fixed_im_sitk=pyr['fixed_im_s'+str(stage)+'_sitk'],
                                        moved_im_affine_sitk=pyr[input_regnet_moving],
                                        fixed_torso_sitk=pyr['fixed_torso_s'+str(stage)+'_sitk'],
                                        moved_torso_affine_sitk=pyr[input_regnet_moving_torso]
                                        )

        # building and loading network
        tf.reset_default_graph()
        setting['R'] = network_dict['Stage'+str(stage)]['R']    # Radius of normal resolution patch size. Total size is (2*R +1)
        setting['Ry'] = network_dict['Stage'+str(stage)]['Ry']  # Radius of output. Total size is (2*Ry +1)
        setting['Ry_erode'] = network_dict['Stage'+str(stage)]['Ry_erode']  # at the test time, sometimes there are some problems at the border
        images_tf = tf.placeholder(tf.float32,
                                   shape=[None, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2 * setting['R'] + 1, 2],
                                   name="Images")
        bn_training = tf.placeholder(tf.bool, name='bn_training')
        x_fixed = images_tf[:, :, :, :, 0, np.newaxis]
        x_deformed = images_tf[:, :, :, :, 1, np.newaxis]
        dvf_tf = getattr(RegNetModel, network_dict['Stage'+str(stage)]['NetworkDesign'])(x_fixed, x_deformed, bn_training)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logging.debug(' Total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, su.address_generator(setting, 'saved_model_with_step',
                                                 current_experiment=network_dict['Stage'+str(stage)]['NetworkLoad'],
                                                 step=network_dict['Stage'+str(stage)]['GlobalStepLoad']))
        while not pair_pyramid.get_sweep_completed():
            # The pyr[DVF_S] is the numpy DVF which will be filled. the dvf_np is an output
            # path from the network. We control the spatial location of both dvf in this function
            batch_im, win_center, win_r_before, win_r_after, predicted_begin, predicted_end = pair_pyramid.next_sweep_patch()
            time_before_gpu = time.time()
            [dvf_np] = sess.run([dvf_tf], feed_dict={images_tf: batch_im, bn_training: 0})
            time_after_gpu = time.time()
            logging.debug('GPU: Data='+pair_info[0]['data']+' CN = {} center = {} is done in {:.2f}s '.format(
                pair_info[0]['cn'], win_center, time_after_gpu - time_before_gpu))

            pyr['DVF_s'+str(stage)][win_center[0] - win_r_before[0]: win_center[0] + win_r_after[0],
                                    win_center[1] - win_r_before[1]: win_center[1] + win_r_after[1],
                                    win_center[2] - win_r_before[2]: win_center[2] + win_r_after[2], :] = \
                dvf_np[0, predicted_begin[0]:predicted_end[0], predicted_begin[1]:predicted_end[1], predicted_begin[2]:predicted_end[2], :]

        pyr['DVF_s'+str(stage)+'_sitk'] = ip.array_to_sitk(pyr['DVF_s' + str(stage)],
                                                           im_ref=pyr['fixed_im_s'+str(stage)+'_sitk'],
                                                           is_vector=True)

        if i_stage == (len(setting['ImagePyramidSchedule'])-1):
            # when all stages are finished, final dvf and moved image are written
            dvf_composed_final_sitk = 'DVF_s'+str(stage)+'_composed_sitk'
            if len(setting['ImagePyramidSchedule']) == 1:
                pyr[dvf_composed_final_sitk] = pyr['DVF_s'+str(stage)+'_sitk']
            else:
                pyr[dvf_composed_final_sitk] = sitk.Add(pyr['DVF_s'+str(setting['ImagePyramidSchedule'][-2])+'_composed_up_sitk'],
                                                        pyr['DVF_s'+str(stage)+'_sitk'])
            sitk.WriteImage(sitk.Cast(pyr[dvf_composed_final_sitk], sitk.sitkVectorFloat32),
                            su.address_generator(setting, 'dvf_s0', pair_info=pair_info, stage_list=stage_list))
            dvf_t = sitk.DisplacementFieldTransform(pyr[dvf_composed_final_sitk])
            pyr['moved_im_s0_sitk'] = ip.resampler_by_dvf(pyr['moving_im_s'+str(stage)+'_sitk'], dvf_t,
                                                          default_pixel_value=setting['data'][pair_info[1]['data']]['defaultPixelValue'])
            sitk.WriteImage(sitk.Cast(pyr['moved_im_s0_sitk'],
                                      setting['data'][pair_info[1]['data']]['imageByte']),
                            su.address_generator(setting, 'moved_image', pair_info=pair_info, stage=0, stage_list=stage_list))
    time_after_dvf = time.time()
    logging.debug('Data='+pair_info[0]['data']+' CN = {} ImType = {} is done in {:.2f}s '.format(
        pair_info[0]['cn'], pair_info[0]['type_im'],time_after_dvf - time_before_dvf))

    return 1
