import os
import time
import SimpleITK as sitk
import numpy as np
import logging
import functions.image.image_processing as ip
import functions.setting.setting_utils as su


def get_dvf_and_deformed_images_seq(setting, im_info=None, stage_sequence=None, mode_synthetic_dvf='generation'):
    """
    This function generates the synthetic displacement vector fields and writes them to disk and returns them.
    If synthetic DVFs are already generated and can be found on the disk, this function just reads and returns them.
    Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction.

    :param setting:
    :param im_info:
                    'data':
                    'deform_exp':
                    'type_im':
                    'CN":            Case Number: (Image Number )Please note that it starts from 1 not 0
                    'dsmooth':       This variable is used to generate another deformed version of the moving image.
                                     Then, use that image to make synthetic DVFs. More information available on [sokooti2017nonrigid]
                    'deform_method'
                    'deform_number'
    :param stage_sequence
    :param mode_synthetic_dvf:      'generation': generating images
                                    'reading':  : only reading images without generating them. In this mode, the function will just wait

    :return im             image with mentioned ImageType and CN
    :return deformed_im    Deformed image by applying the synthetic DeformedDVF_ on the Im_
    :return dvf            Syntethic DVF
    """
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'padto': im_info['padto']}
    # A dictionary of arguments, which only be used in setting_utils.address_generator. This dictionary helps to have shorter code line.
    logging.debug('artifical_generation[' + mode_synthetic_dvf + ']: Data=' + im_info['data'] + ', TypeIm=' + str(im_info['type_im']) + ', CN=' + str(im_info['cn']) +
                  ' Dsmooth=' + str(im_info['dsmooth']) + ' D=' + str(im_info['deform_number']) +
                  '  starting')
    time_before = time.time()
    if stage_sequence != [4, 2, 1]:
        raise ValueError(' not implemented ')

    # later
    # for stage in stage_sequence:
    #     if stage != 1:
    #         check_downsampled_images(setting, im_info, stage)
    mask_to_zero = setting['deform_exp'][im_info['deform_exp']]['MaskToZero']

    # this is the last image to write by registration:
    if setting['UseRegisteredImages']:
        moved_mask_s1 = su.address_generator(setting,'Moved'+mask_to_zero+'_AG', stage=1, **im_info_su)

        while not os.path.isfile(moved_mask_s1):
            logging.debug('artifical_generation[' + mode_synthetic_dvf + ']: Data=' + im_info['data'] + ', TypeIm=' + str(im_info['type_im']) + ', CN=' + str(im_info['cn']) +
                          ' Dsmooth=' + str(im_info['dsmooth']) + ' D=' + str(im_info['deform_number']) +
                          ' waiting for moved images by registration')
            time.sleep(5)
        if (time.time() - os.path.getmtime(moved_mask_s1)) < 5:
            time.sleep(5)
    else:
        check_downsampled_images(setting, im_info, stage=2)
        check_downsampled_images(setting, im_info, stage=4)

    fixed_im_seq = dict()
    fixed_mask_seq = dict()
    moved_im_seq = dict()
    moved_mask_seq = dict()

    for stage in stage_sequence:
        fixed_im_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
            setting, 'DeformedIm', deformed_im_ext=im_info['deformed_im_ext'], stage=stage, **im_info_su)))
        fixed_mask_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
            setting, 'Deformed'+mask_to_zero, deformed_im_ext=im_info['deformed_im_ext'], stage=stage, **im_info_su)))

    moved_im_seq['stage4'] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
        setting, 'Im', stage=4, **im_info_su)))
    moved_mask_seq['stage4'] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
        setting, mask_to_zero, stage=4, **im_info_su)))
    for stage in [2, 1]:
        if setting['UseRegisteredImages']:
            moved_im_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
                setting, 'MovedIm_AG', stage=stage, **im_info_su)))
            moved_mask_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
                setting, 'Moved'+mask_to_zero+'_AG', stage=stage, **im_info_su)))
        else:
            moved_im_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
                setting, 'Im', stage=stage, **im_info_su)))
            moved_mask_seq['stage'+str(stage)] = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
                setting, mask_to_zero, stage=stage, **im_info_su)))

    if mask_to_zero is not None:
        for stage in stage_sequence:
            size_fixed_im = np.array(np.shape(fixed_im_seq['stage'+str(stage)]))
            size_fixed_mask = np.array(np.shape(fixed_mask_seq['stage'+str(stage)]))
            size_diff = size_fixed_im - size_fixed_mask
            if abs(sum(size_diff)):
                if sum(size_diff) > 0:
                    # mask is smaller so we need to pad it
                    fixed_mask_seq['stage'+str(stage)] = np.pad(fixed_mask_seq['stage'+str(stage)],
                                                                ((0, size_diff[0]),
                                                                 (0, size_diff[1]),
                                                                 (0, size_diff[2])),
                                                                'constant', constant_values=(0,))
                else:
                    # mask is bigger
                    size_mask_crop = size_fixed_mask + size_diff
                    fixed_mask_seq['stage'+str(stage)] = fixed_mask_seq['stage'+str(stage)][0:size_mask_crop[0], 0:size_mask_crop[1], 0:size_mask_crop[2]]

            size_moved_im = np.array(np.shape(moved_im_seq['stage'+str(stage)]))
            size_moved_mask = np.array(np.shape(moved_mask_seq['stage'+str(stage)]))
            size_diff = size_moved_im - size_moved_mask
            if abs(sum(size_diff)):
                logging.debug('warning size are not same.............. size_dife={}'.format(size_diff))
                if sum(size_diff) > 0:
                    # mask is smaller so we need to pad it
                    moved_mask_seq['stage'+str(stage)] = np.pad(moved_mask_seq['stage'+str(stage)],
                                                                ((0, size_diff[0]),
                                                                 (0, size_diff[1]),
                                                                 (0, size_diff[2])),
                                                                'constant', constant_values=(0,))
                else:
                    # mask is bigger
                    moved_mask_seq['stage'+str(stage)] = moved_mask_seq['stage'+str(stage)][0:size_diff[0], 0:size_diff[1], 0:size_diff[2]]

            fixed_im_seq['stage'+str(stage)][fixed_mask_seq['stage'+str(stage)] == 0] = setting['data'][im_info['data']]['DefaultPixelValue']
            moved_im_seq['stage'+str(stage)][moved_mask_seq['stage'+str(stage)] == 0] = setting['data'][im_info['data']]['DefaultPixelValue']

    for stage in stage_sequence:
        # stage 4 and 2 has more padding in order to be possible that all patches have the same center
        if setting['ImPad_S'+str(stage)] > 0:
            fixed_im_seq['stage'+str(stage)] = np.pad(fixed_im_seq['stage'+str(stage)], setting['ImPad_S'+str(stage)], 'constant',
                                                      constant_values=(setting['data'][im_info['data']]['DefaultPixelValue'],))
            moved_im_seq['stage'+str(stage)] = np.pad(moved_im_seq['stage'+str(stage)], setting['ImPad_S'+str(stage)], 'constant',
                                                      constant_values=(setting['data'][im_info['data']]['DefaultPixelValue'],))

    dvf_threshold_list = setting['DVFThresholdList']
    dvf_label_address = su.address_generator(setting, 'DeformedDVFLabel', stage=1, dvf_threshold_list=dvf_threshold_list, **im_info_su)
    if not os.path.isfile(dvf_label_address):
        dvf_address = su.address_generator(setting, 'DeformedDVF', stage=1, **im_info_su)
        dvf = sitk.GetArrayFromImage(sitk.ReadImage(dvf_address))
        dvf_magnitude = np.sqrt(np.square(dvf[:, :, :, 0]) + np.square(dvf[:, :, :, 1]) + np.square(dvf[:, :, :, 2]))
        dvf_label = np.zeros(dvf_magnitude.shape, dtype=np.int8)
        for i_label in range(len(dvf_threshold_list)-1):
            dvf_label[(dvf_threshold_list[i_label] <= dvf_magnitude) & (dvf_magnitude < dvf_threshold_list[i_label+1])] = i_label

        dvf_label_sitk = ip.array_to_sitk(dvf_label, im_ref=sitk.ReadImage(su.address_generator(
            setting, 'DeformedIm', deformed_im_ext=im_info['deformed_im_ext'], stage=stage, **im_info_su)))
        sitk.WriteImage(sitk.Cast(dvf_label_sitk, sitk.sitkInt8), dvf_label_address)
    else:
        dvf_label = sitk.GetArrayFromImage(sitk.ReadImage(dvf_label_address))

    # no mask for now
    # dvf_mask = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(
    #     setting, 'Deformed' + setting['Label_Mask'], deformed_im_ext=im_info['deformed_im_ext'], stage=stage, **im_info_su)))
    # dvf_label[dvf_mask == 0] = 0

    time_after = time.time()
    logging.debug('artifical_generation[' + mode_synthetic_dvf + ']: Data=' + im_info['data'] + ', TypeIm=' + str(im_info['type_im']) + ', CN=' + str(im_info['cn']) +
                  ' Dsmooth=' + str(im_info['dsmooth']) + ' D=' + str(im_info['deform_number']) +
                  ' is Done in {:.3f}s'.format(time_after - time_before))
    return fixed_im_seq, moved_im_seq, dvf_label, fixed_mask_seq


def wait_for_image(im_address, print_address=None):
    if print_address is None:
        print_address = im_address
    count_wait = 1
    while not os.path.isfile(im_address):
        time.sleep(5)
        logging.debug('artifical_generation[reading]: waiting {}s for '.format(count_wait * 5) + print_address)
        count_wait += 1
    return 0


def check_downsampled_images(setting, im_info, stage):
    padto = im_info['padto']
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth']}

    im_list_downsample = [{'Image': 'Im',
                           'interpolator': 'BSpline',
                           'DefaultPixelValue': setting['data'][im_info['data']]['DefaultPixelValue'],
                           'ImageByte': setting['data'][im_info['data']]['ImageByte']},

                          {'Image': 'Lung',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8},

                          {'Image': 'Torso',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8},

                          {'Image': 'DeformedIm',
                           'interpolator': 'BSpline',
                           'DefaultPixelValue': setting['data'][im_info['data']]['DefaultPixelValue'],
                           'ImageByte': setting['data'][im_info['data']]['ImageByte']},

                          {'Image': 'DeformedLung',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8},

                          {'Image': 'DeformedTorso',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8}

                          ]

    for im_dict in im_list_downsample:
        im_stage_address = su.address_generator(setting, im_dict['Image'], stage=stage, padto=padto,
                                                deformed_im_ext=im_info['deformed_im_ext'], **im_info_su)
        # remove the second condition later, that was for fixing some images ['Im', 'Lung'] --> I removed

        if not os.path.isfile(im_stage_address):
            im_s1_sitk = sitk.ReadImage(su.address_generator(setting, im_dict['Image'], stage=1, padto=padto,
                                                             deformed_im_ext=im_info['deformed_im_ext'], **im_info_su))
            if setting['DownSamplingByGPU'] and im_dict['Image'] == 'OriginalIm':
                im_s1 = sitk.GetArrayFromImage(im_s1_sitk)
                im_stage = ip.downsampler_gpu(im_s1, stage, normalize_kernel=True,
                                              default_pixel_value=im_dict['DefaultPixelValue'])

                im_stage_sitk = ip.array_to_sitk(im_stage,
                                                 origin=im_s1_sitk.GetOrigin(),
                                                 spacing=tuple(i * stage for i in im_s1_sitk.GetSpacing()),
                                                 direction=im_s1_sitk.GetDirection())
            else:
                if im_dict['interpolator'] == 'NearestNeighbor':
                    interpolator = sitk.sitkNearestNeighbor
                elif im_dict['interpolator'] == 'BSpline':
                    interpolator = sitk.sitkBSpline
                else:
                    raise ValueError("interpolator should be in ['NearestNeighbor', 'BSpline']")

                if im_dict['Image'] in ['Torso', 'Lung']:
                    im_ref_sitk = sitk.ReadImage(su.address_generator(setting, 'Im', stage=stage, padto=padto, **im_info_su))
                else:
                    im_ref_sitk = None
                im_stage_sitk = ip.resampler_sitk(im_s1_sitk,
                                                  scale=stage,
                                                  im_ref=im_ref_sitk,
                                                  default_pixel_value=im_dict['DefaultPixelValue'],
                                                  interpolator=interpolator)

                # for debugging
                # sitk.WriteImage(sitk.Cast(im_stage_sitk, im_dict['ImageByte']),
                #                 su.address_generator(setting, im_dict['Image'], stage=stage, padto=None, **im_info_su))
                if padto is not None:
                    dim_im = setting['Dim']
                    pad_before = np.zeros(dim_im, dtype=np.int)
                    pad_after = np.zeros(dim_im, dtype=np.int)
                    im_size = np.array(im_stage_sitk.GetSize())
                    extra_to_pad = padto - im_size
                    for d in range(dim_im):
                        if extra_to_pad[d] < 0:
                            raise ValueError('size of the padto='+str(padto)+' should be smaller than the size of the image {}'.format(im_size))
                        elif extra_to_pad[d] == 0:
                            pad_before[d] = 0
                            pad_after[d] = 0
                        else:
                            if extra_to_pad[d] % 2 == 0:
                                pad_before[d] = np.int(extra_to_pad[d] / 2)
                                pad_after[d] = np.int(extra_to_pad[d] / 2)
                            else:
                                pad_before[d] = np.floor(extra_to_pad[d] / 2)
                                pad_after[d] = np.floor(extra_to_pad[d] / 2) + 1

                    pad_before = [int(p) for p in pad_before]
                    pad_after = [int(p) for p in pad_after]
                    im_stage_sitk = sitk.ConstantPad(im_stage_sitk,
                                                     [pad_before[0], pad_before[1], pad_before[2]],
                                                     [pad_after[0], pad_after[1], pad_after[2]],
                                                     constant=im_dict['DefaultPixelValue'],)

            sitk.WriteImage(sitk.Cast(im_stage_sitk, im_dict['ImageByte']), im_stage_address)
    return 0
