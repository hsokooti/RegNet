import logging
import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf
import time
import functions.image.image_processing as ip
import functions.network as network
import functions.reading.real_pair as real_pair
import functions.setting.setting_utils as su
import functions.tf_utils as tfu


def multi_stage(setting, pair_info, overwrite=False):
    """
    :param setting:
    :param pair_info: information of the pair to be registered.
    :param overwrite:
    :return: The output moved images and dvf will be written to the disk.
             1: registration is performed correctly
             2: skip overwriting
             3: the dvf is available from the previous experiment [4, 2, 1]. Then just upsample it.
    """
    stage_list = setting['ImagePyramidSchedule']
    if setting['read_pair_mode'] == 'synthetic':
        deformed_im_ext = pair_info[0].get('deformed_im_ext', None)
        im_info_su = {'data': pair_info[0]['data'], 'deform_exp': pair_info[0]['deform_exp'], 'type_im': pair_info[0]['type_im'],
                      'cn': pair_info[0]['cn'], 'dsmooth': pair_info[0]['dsmooth'], 'padto': pair_info[0]['padto'],
                      'deformed_im_ext': deformed_im_ext}
        moved_im_s0_address = su.address_generator(setting, 'MovedIm_AG', stage=1, **im_info_su)
        moved_torso_s1_address = su.address_generator(setting, 'MovedTorso_AG', stage=1, **im_info_su)
        moved_lung_s1_address = su.address_generator(setting, 'MovedLung_AG', stage=1, **im_info_su)
    else:
        moved_im_s0_address = su.address_generator(setting, 'MovedIm', pair_info=pair_info, stage=0, stage_list=stage_list)
        moved_torso_s1_address = None
        moved_lung_s1_address = None

    if setting['read_pair_mode'] == 'synthetic':
        if os.path.isfile(moved_im_s0_address) and os.path.isfile(moved_torso_s1_address):
            if not overwrite:
                logging.debug('overwrite=False, file '+moved_im_s0_address+' already exists, skipping .....')
                return 2
            else:
                logging.debug('overwrite=True, file '+moved_im_s0_address+' already exists, but overwriting .....')
    else:
        if os.path.isfile(moved_im_s0_address):
            if not overwrite:
                logging.debug('overwrite=False, file ' + moved_im_s0_address + ' already exists, skipping .....')
                return 2
            else:
                logging.debug('overwrite=True, file ' + moved_im_s0_address + ' already exists, but overwriting .....')

    pair_stage1 = real_pair.Images(setting, pair_info, stage=1, padto=setting['PadTo']['stage1']) # just read the original images without any padding
    pyr = dict()  # pyr: a dictionary of pyramid images
    pyr['fixed_im_s1_sitk'] = pair_stage1.get_fixed_im_sitk()
    pyr['moving_im_s1_sitk'] = pair_stage1.get_moved_im_affine_sitk()
    pyr['fixed_im_s1'] = pair_stage1.get_fixed_im()
    pyr['moving_im_s1'] = pair_stage1.get_moved_im_affine()
    if setting['UseMask']:
        pyr['fixed_mask_s1_sitk'] = pair_stage1.get_fixed_mask_sitk()
        pyr['moving_mask_s1_sitk'] = pair_stage1.get_moved_mask_affine_sitk()
    if setting['read_pair_mode'] == 'real':
        if not (os.path.isdir(su.address_generator(setting, 'full_reg_folder', pair_info=pair_info, stage_list=stage_list))):
            os.makedirs(su.address_generator(setting, 'full_reg_folder', pair_info=pair_info, stage_list=stage_list))
    setting['GPUMemory'], setting['NumberOfGPU'] = tfu.client.read_gpu_memory()
    time_before_dvf = time.time()

    # check if DVF is available from the previous experiment [4, 2, 1]. Then just upsample it.
    if stage_list in [[4, 2], [4]]:
        dvf0_address = su.address_generator(setting, 'dvf_s0', pair_info=pair_info, stage_list=stage_list)
        chosen_stage = None
        if stage_list == [4, 2]:
            chosen_stage = 2
        elif stage_list == [4]:
            chosen_stage = 4
        if chosen_stage is not None:
            dvf_s_up_address = su.address_generator(setting, 'dvf_s_up', pair_info=pair_info, stage=chosen_stage, stage_list=[4, 2, 1])
            if os.path.isfile(dvf_s_up_address):
                logging.debug('DVF found from prev exp:' + dvf_s_up_address + ', only performing upsampling')
                dvf_s_up = sitk.ReadImage(dvf_s_up_address)
                dvf0 = ip.resampler_sitk(dvf_s_up,
                                         scale=1 / (chosen_stage / 2),
                                         im_ref_size=pyr['fixed_im_s1_sitk'].GetSize(),
                                         interpolator=sitk.sitkLinear
                                         )
                sitk.WriteImage(sitk.Cast(dvf0, sitk.sitkVectorFloat32), dvf0_address)
                return 3

    for i_stage, stage in enumerate(setting['ImagePyramidSchedule']):
        mask_to_zero_stage = setting['network_dict']['stage' + str(stage)]['MaskToZero']
        if stage != 1:
            pyr['fixed_im_s'+str(stage)+'_sitk'] = ip.downsampler_gpu(pyr['fixed_im_s1_sitk'], stage,
                                                                      default_pixel_value=setting['data'][pair_info[0]['data']]['DefaultPixelValue'])
            pyr['moving_im_s'+str(stage)+'_sitk'] = ip.downsampler_gpu(pyr['moving_im_s1_sitk'], stage,
                                                                       default_pixel_value=setting['data'][pair_info[1]['data']]['DefaultPixelValue'])
        if setting['UseMask']:
            pyr['fixed_mask_s'+str(stage)+'_sitk'] = ip.resampler_sitk(pyr['fixed_mask_s1_sitk'],
                                                                       scale=stage,
                                                                       im_ref=pyr['fixed_im_s' + str(stage) + '_sitk'],
                                                                       default_pixel_value=0,
                                                                       interpolator=sitk.sitkNearestNeighbor)
            pyr['moving_mask_s'+str(stage)+'_sitk'] = ip.resampler_sitk(pyr['moving_mask_s1_sitk'],
                                                                        scale=stage,
                                                                        im_ref=pyr['moving_im_s' + str(stage) + '_sitk'],
                                                                        default_pixel_value=0,
                                                                        interpolator=sitk.sitkNearestNeighbor)

            if setting['WriteMasksForLSTM']:
                # only to be used in sequential training (LSTM)
                if setting['read_pair_mode'] == 'synthetic':
                    fixed_mask_stage_address = su.address_generator(setting, 'Deformed'+mask_to_zero_stage, stage=stage, **im_info_su)
                    moving_mask_stage_address = su.address_generator(setting, mask_to_zero_stage, stage=stage, **im_info_su)
                    fixed_im_stage_address = su.address_generator(setting, 'DeformedIm', stage=stage, **im_info_su)
                    sitk.WriteImage(sitk.Cast(pyr['fixed_im_s' + str(stage) + '_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']), fixed_im_stage_address)
                    sitk.WriteImage(pyr['fixed_mask_s'+str(stage)+'_sitk'], fixed_mask_stage_address)
                    if im_info_su['dsmooth'] != 0 and stage == 4:
                        # not overwirte original images
                        moving_im_stage_address = su.address_generator(setting, 'Im', stage=stage, **im_info_su)
                        sitk.WriteImage(sitk.Cast(pyr['moving_im_s' + str(stage) + '_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']), moving_im_stage_address)
                        sitk.WriteImage(pyr['moving_mask_s' + str(stage) + '_sitk'], moving_mask_stage_address)
                else:
                    fixed_im_stage_address = su.address_generator(setting, 'Im', stage=stage, **pair_info[0])
                    fixed_mask_stage_address = su.address_generator(setting, mask_to_zero_stage, stage=stage, **pair_info[0])
                    if not os.path.isfile(fixed_im_stage_address):
                        sitk.WriteImage(sitk.Cast(pyr['fixed_im_s'+str(stage)+'_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']), fixed_im_stage_address)
                    if not os.path.isfile(fixed_mask_stage_address):
                        sitk.WriteImage(pyr['fixed_mask_s'+str(stage)+'_sitk'], fixed_mask_stage_address)
                    if i_stage == 0:
                        moved_im_affine_stage_address = su.address_generator(setting, 'MovedImBaseReg', pair_info=pair_info, stage=stage, **pair_info[1])
                        moved_mask_affine_stage_address = su.address_generator(setting, 'Moved'+mask_to_zero_stage+'BaseReg', pair_info=pair_info, stage=stage, **pair_info[1])
                        if not os.path.isfile(moved_im_affine_stage_address):
                            sitk.WriteImage(sitk.Cast(pyr['moving_im_s'+str(stage)+'_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']), moved_im_affine_stage_address)
                        if not os.path.isfile(moved_mask_affine_stage_address):
                            sitk.WriteImage(pyr['moving_mask_s'+str(stage)+'_sitk'], moved_mask_affine_stage_address)

        else:
            pyr['fixed_mask_s'+str(stage)+'_sitk'] = None
            pyr['moving_mask_s'+str(stage)+'_sitk'] = None
        input_regnet_moving_mask = None
        if i_stage == 0:
            input_regnet_moving = 'moving_im_s'+str(stage)+'_sitk'
            if setting['UseMask']:
                input_regnet_moving_mask = 'moving_mask_s'+str(stage)+'_sitk'
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
                                                                  output_shape_3d=pyr['fixed_im_s'+str(stage)+'_sitk'].GetSize()[::-1],
                                                                  )
            if setting['WriteAfterEachStage'] and not setting['WriteNoDVF']:
                sitk.WriteImage(sitk.Cast(pyr[dvf_composed_previous_up_sitk], sitk.sitkVectorFloat32),
                                su.address_generator(setting, 'dvf_s_up', pair_info=pair_info, stage=previous_pyramid, stage_list=stage_list))

            dvf_t = sitk.DisplacementFieldTransform(pyr[dvf_composed_previous_up_sitk])
            # after this line DVF_composed_previous_up_sitk is converted to a transform. so we need to load it again.
            pyr['moved_im_s'+str(stage)+'_sitk'] = ip.resampler_by_transform(pyr['moving_im_s' + str(stage) + '_sitk'],
                                                                             dvf_t,
                                                                             default_pixel_value=setting['data'][pair_info[1]['data']]['DefaultPixelValue'])
            if setting['UseMask']:
                pyr['moved_mask_s'+str(stage)+'_sitk'] = ip.resampler_by_transform(pyr['moving_mask_s' + str(stage) + '_sitk'],
                                                                                   dvf_t,
                                                                                   default_pixel_value=0,
                                                                                   interpolator=sitk.sitkNearestNeighbor)

            pyr[dvf_composed_previous_up_sitk] = dvf_t.GetDisplacementField()
            if setting['WriteAfterEachStage']:
                if setting['read_pair_mode'] == 'synthetic':
                    moved_im_s_address = su.address_generator(setting, 'MovedIm_AG', stage=stage, **im_info_su)
                    moved_mask_s_address = su.address_generator(setting, 'Moved'+mask_to_zero_stage+'_AG', stage=stage, **im_info_su)
                else:
                    moved_im_s_address = su.address_generator(setting, 'MovedIm', pair_info=pair_info, stage=stage, stage_list=stage_list)
                    moved_mask_s_address = su.address_generator(setting, 'Moved'+mask_to_zero_stage, pair_info=pair_info, stage=stage, stage_list=stage_list)

                sitk.WriteImage(sitk.Cast(pyr['moved_im_s'+str(stage)+'_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']),
                                moved_im_s_address)

                if setting['WriteMasksForLSTM']:
                    sitk.WriteImage(pyr['moved_mask_s' + str(stage) + '_sitk'], moved_mask_s_address)

            input_regnet_moving = 'moved_im_s'+str(stage)+'_sitk'
            if setting['UseMask']:
                input_regnet_moving_mask = 'moved_mask_s'+str(stage)+'_sitk'

        pyr['DVF_s'+str(stage)] = np.zeros(np.r_[pyr['fixed_im_s'+str(stage)+'_sitk'].GetSize()[::-1], 3], dtype=np.float64)
        if setting['network_dict']['stage'+str(stage)]['R'] == 'Auto' and \
                setting['network_dict']['stage'+str(stage)]['Ry'] == 'Auto':
                current_network_name = setting['network_dict']['stage'+str(stage)]['NetworkDesign']
                r_out_erode_default = setting['network_dict']['stage' + str(stage)]['Ry_erode']
                r_in, r_out, r_out_erode = network.utils.find_optimal_radius(pyr['fixed_im_s'+str(stage)+'_sitk'],
                                                                             current_network_name, r_out_erode_default,
                                                                             gpu_memory=setting['GPUMemory'],
                                                                             number_of_gpu=setting['NumberOfGPU'])

        else:
            r_in = setting['network_dict']['stage'+str(stage)]['R']    # Radius of normal resolution patch size. Total size is (2*R +1)
            r_out = setting['network_dict']['stage'+str(stage)]['Ry']  # Radius of output. Total size is (2*Ry +1)
            r_out_erode = setting['network_dict']['stage'+str(stage)]['Ry_erode']  # at the test time, sometimes there are some problems at the border

        logging.debug('stage'+str(stage)+' ,'+pair_info[0]['data']+', CN{}, ImType{}, Size={}'.
                      format(pair_info[0]['cn'], pair_info[0]['type_im'],
                      pyr['fixed_im_s'+str(stage)+'_sitk'].GetSize()[::-1])+', ' +
                      setting['network_dict']['stage'+str(stage)]['NetworkDesign']+': r_in:{}, r_out:{}, r_out_erode:{}'.
                      format(r_in, r_out, r_out_erode))
        pair_pyramid = real_pair.Images(setting, pair_info, stage=stage,
                                        fixed_im_sitk=pyr['fixed_im_s'+str(stage)+'_sitk'],
                                        moved_im_affine_sitk=pyr[input_regnet_moving],
                                        fixed_mask_sitk=pyr['fixed_mask_s' + str(stage) + '_sitk'],
                                        moved_mask_affine_sitk=pyr[input_regnet_moving_mask],
                                        padto=setting['PadTo']['stage'+str(stage)],
                                        r_in=r_in,
                                        r_out=r_out,
                                        r_out_erode=r_out_erode
                                        )

        # building and loading network
        tf.reset_default_graph()
        images_tf = tf.placeholder(tf.float32,
                                   shape=[None, 2*r_in+1, 2*r_in+1, 2*r_in+1, 2],
                                   name="Images")
        bn_training = tf.placeholder(tf.bool, name='bn_training')
        dvf_tf = getattr(getattr(network, setting['network_dict']['stage'+str(stage)]['NetworkDesign']), 'network')(images_tf, bn_training)
        logging.debug(' Total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        sess = tf.Session()
        saver.restore(sess, su.address_generator(setting, 'saved_model_with_step',
                                                 current_experiment=setting['network_dict']['stage'+str(stage)]['NetworkLoad'],
                                                 step=setting['network_dict']['stage'+str(stage)]['GlobalStepLoad']))
        while not pair_pyramid.get_sweep_completed():
            # The pyr[DVF_S] is the numpy DVF which will be filled. the dvf_np is an output
            # patch from the network. We control the spatial location of both dvf in this function
            batch_im, win_center, win_r_before, win_r_after, predicted_begin, predicted_end = pair_pyramid.next_sweep_patch()
            time_before_gpu = time.time()
            [dvf_np] = sess.run([dvf_tf], feed_dict={images_tf: batch_im, bn_training: 0})
            time_after_gpu = time.time()
            logging.debug('GPU: '+pair_info[0]['data']+', CN{} center={} is done in {:.2f}s '.format(
                pair_info[0]['cn'], win_center, time_after_gpu - time_before_gpu))

            pyr['DVF_s'+str(stage)][win_center[0] - win_r_before[0]: win_center[0] + win_r_after[0],
                                    win_center[1] - win_r_before[1]: win_center[1] + win_r_after[1],
                                    win_center[2] - win_r_before[2]: win_center[2] + win_r_after[2], :] = \
                dvf_np[0, predicted_begin[0]:predicted_end[0], predicted_begin[1]:predicted_end[1], predicted_begin[2]:predicted_end[2], :]
            # rescaling dvf based on the voxel spacing:
            spacing_ref = [1.0 * stage for _ in range(3)]
            spacing_current = pyr['fixed_im_s' + str(stage) + '_sitk'].GetSpacing()
            for dim in range(3):
                pyr['DVF_s' + str(stage)][:, :, :, dim] = pyr['DVF_s' + str(stage)][:, :, :, dim] * spacing_current[dim] / spacing_ref[dim]

        pyr['DVF_s'+str(stage)+'_sitk'] = ip.array_to_sitk(pyr['DVF_s' + str(stage)],
                                                           im_ref=pyr['fixed_im_s'+str(stage)+'_sitk'],
                                                           is_vector=True)

        if i_stage == (len(setting['ImagePyramidSchedule'])-1):
            # when all stages are finished, final dvf and moved image are written
            dvf_composed_final_sitk = 'DVF_s'+str(stage)+'_composed_sitk'
            if len(setting['ImagePyramidSchedule']) == 1:
                # need to upsample in the case that last stage is not 1
                if stage == 1:
                    pyr[dvf_composed_final_sitk] = pyr['DVF_s'+str(stage)+'_sitk']
                else:
                    pyr[dvf_composed_final_sitk] = ip.resampler_sitk(pyr['DVF_s'+str(stage)+'_sitk'],
                                                                     scale=1/stage,
                                                                     im_ref_size=pyr['fixed_im_s1_sitk'].GetSize(),
                                                                     interpolator=sitk.sitkLinear
                                                                     )
            else:
                pyr[dvf_composed_final_sitk] = sitk.Add(pyr['DVF_s'+str(setting['ImagePyramidSchedule'][-2])+'_composed_up_sitk'],
                                                        pyr['DVF_s'+str(stage)+'_sitk'])
                if stage != 1:
                    pyr[dvf_composed_final_sitk] = ip.resampler_sitk(pyr[dvf_composed_final_sitk],
                                                                     scale=1/stage,
                                                                     im_ref_size=pyr['fixed_im_s1_sitk'].GetSize(),
                                                                     interpolator=sitk.sitkLinear
                                                                     )
            if not setting['WriteNoDVF']:
                sitk.WriteImage(sitk.Cast(pyr[dvf_composed_final_sitk], sitk.sitkVectorFloat32),
                                su.address_generator(setting, 'dvf_s0', pair_info=pair_info, stage_list=stage_list))
            dvf_t = sitk.DisplacementFieldTransform(pyr[dvf_composed_final_sitk])
            pyr['moved_im_s0_sitk'] = ip.resampler_by_transform(pyr['moving_im_s1_sitk'], dvf_t,
                                                                default_pixel_value=setting['data'][pair_info[1]['data']]['DefaultPixelValue'])
            sitk.WriteImage(sitk.Cast(pyr['moved_im_s0_sitk'], setting['data'][pair_info[1]['data']]['ImageByte']), moved_im_s0_address)

            if setting['WriteMasksForLSTM']:
                mask_to_zero_stage = setting['network_dict']['stage' + str(stage)]['MaskToZero']
                if setting['read_pair_mode'] == 'synthetic':
                    moving_mask_sitk = sitk.ReadImage(su.address_generator(setting, mask_to_zero_stage, stage=1, **im_info_su))
                    moved_mask_stage1 = ip.resampler_by_transform(moving_mask_sitk,
                                                                  dvf_t,
                                                                  default_pixel_value=0,
                                                                  interpolator=sitk.sitkNearestNeighbor)
                    sitk.WriteImage(moved_mask_stage1, su.address_generator(setting, 'Moved'+mask_to_zero_stage+'_AG', stage=1, **im_info_su))
                    logging.debug('writing '+ su.address_generator(setting, 'Moved'+mask_to_zero_stage+'_AG', stage=1, **im_info_su))

    time_after_dvf = time.time()
    logging.debug(pair_info[0]['data']+', CN{}, ImType{} is done in {:.2f}s '.format(
        pair_info[0]['cn'], pair_info[0]['type_im'], time_after_dvf - time_before_dvf))

    return 0


def calculate_jacobian(setting, pair_info, overwrite=False):
    stage_list = setting['ImagePyramidSchedule']
    if setting['current_experiment'] == 'elx_registration':
        dvf_name = 'DVFBSpline'
        jac_name = 'DVFBSpline_Jac'
    else:
        dvf_name = 'dvf_s0'
        jac_name = 'dvf_s0_jac'
    jac_address = su.address_generator(setting, jac_name,  pair_info=pair_info, stage_list=stage_list)

    if overwrite or not os.path.isfile(jac_address):
        time_before_jac = time.time()
        dvf0_address = su.address_generator(setting, dvf_name, pair_info=pair_info, stage_list=stage_list)
        dvf0_sitk = sitk.ReadImage(dvf0_address)
        dvf0 = sitk.GetArrayFromImage(dvf0_sitk)
        spacing = dvf0_sitk.GetSpacing()[::-1]
        jac = ip.calculate_jac(dvf0, spacing)
        sitk.WriteImage(sitk.GetImageFromArray(jac.astype(np.float32)), jac_address)
        time_after_jac = time.time()
        logging.debug(pair_info[0]['data']+', CN{}, ImType{} Jacobian is done in {:.2f}s '.format(
            pair_info[0]['cn'], pair_info[0]['type_im'], time_after_jac - time_before_jac))
    else:
        logging.debug(pair_info[0]['data']+', CN{}, ImType{} Jacobian is already available. skipping... '.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))

        # jac_hist_max = 3
        # jac_hist_min = -1
        # step_h = 0.2
        # if np.max(jac) > jac_hist_max:
        #     jac_hist_max = np.ceil(np.max(jac))
        # if np.min(jac) < jac_hist_min:
        #     jac_hist_min = np.floor(np.min(jac))
        #
        # folding_percentage = np.sum(jac < 0) / np.prod(np.shape(jac)) * 100
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(np.ravel(jac), log=True, bins=np.arange(jac_hist_min, jac_hist_max+step_h, step_h))
        # plt.title('min(Jac)={:.2f}, max(Jac)={:.2f}, folding={:.5f}%'.format(np.min(jac), np.max(jac), folding_percentage))
        # plt.draw()
        # plt.savefig(su.address_generator(setting, 'dvf_s0_jac_hist_plot', pair_info=pair_info, stage_list=stage_list))
        # plt.close()

    # write histograms!
    return 0
