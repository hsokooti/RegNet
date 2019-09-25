from . import dvf_generation
from . import intensity_augmentation
import os
import time
import SimpleITK as sitk
import numpy as np
import logging
import functions.image.image_processing as ip
import functions.setting.setting_utils as su


def get_dvf_and_deformed_images(setting, im_info=None, stage=None, mode_synthetic_dvf='generation'):
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
    :param stage
    :param mode_synthetic_dvf:      'generation': generating images
                                    'reading':  : only reading images without generating them. In this mode, the function will just wait

    :return im             image with mentioned ImageType and CN
    :return deformed_im    Deformed image by applying the synthetic DeformedDVF_ on the Im_
    :return dvf            Syntethic DVF
    """
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    # A dictionary of arguments, which only be used in setting_utils.address_generator. This dictionary helps to have shorter code line.

    if stage > 1:
        check_downsampled_original_images(setting, im_info, stage)

    if im_info['dsmooth'] == 0:
        im_sitk = ip.ReadImage(su.address_generator(setting, 'OriginalIm', **im_info_su))
        if setting['UseTorsoMask']:
            im_torso_sitk = sitk.ReadImage(su.address_generator(setting, 'OriginalTorso', **im_info_su))
        if setting['UseLungMask']:
            im_lung_sitk = sitk.ReadImage(su.address_generator(setting, 'OriginalLung', **im_info_su))
    else:
        next_im_address = (su.address_generator(setting, 'NextIm', **im_info_su))
        if not os.path.isfile(next_im_address):
            if mode_synthetic_dvf == 'reading':
                wait_for_image(next_im_address)
            if mode_synthetic_dvf == 'generation':
                logging.debug('artifical_generation[generation]: start '+su.address_generator(setting, 'NextIm', print_mode=True, **im_info_su))
                generate_next_im(setting, im_info=im_info, stage=stage)
        im_sitk = ip.ReadImage(next_im_address, waiting_time=3)
        if setting['UseTorsoMask']:
            # please note that torso might be created a few minutes after NextIm. So we have to wait for the torso.
            im_torso_address = su.address_generator(setting, 'NextTorso', **im_info_su)
            im_lung_address = su.address_generator(setting, 'NextLung', **im_info_su)
            if mode_synthetic_dvf == 'reading':
                wait_for_image(im_torso_address)
                wait_for_image(im_lung_address)
            im_torso_sitk = ip.ReadImage(im_torso_address, waiting_time=1)
            im_lung_sitk = ip.ReadImage(im_lung_address, waiting_time=1)

    im = sitk.GetArrayFromImage(im_sitk)
    start_time = time.time()
    dvf_address = su.address_generator(setting, 'DeformedDVF', **im_info_su)
    deformed_im_address = su.address_generator(setting, 'DeformedIm', deformed_im_ext=im_info['deformed_im_ext'], **im_info_su)

    if not os.path.isfile(deformed_im_address):
        if mode_synthetic_dvf == 'reading':
            wait_for_image(dvf_address)
            wait_for_image(deformed_im_address)

        if mode_synthetic_dvf == 'generation':
            if os.path.isfile(dvf_address):
                dvf_sitk = sitk.ReadImage(dvf_address)
            else:
                logging.debug('artifical_generation[generation]: start ' +
                              su.address_generator(setting, 'DeformedDVF', print_mode=True, **im_info_su))
                if not os.path.exists(su.address_generator(setting, 'DFolder', **im_info_su)):
                    os.makedirs(su.address_generator(setting, 'DFolder', **im_info_su))

                if setting['DVFPad_S'+str(stage)] > 0:
                    pad = setting['DVFPad_S'+str(stage)]
                    dvf_ref_sitk = sitk.ConstantPad(im_sitk, [pad, pad, pad], [pad, pad, pad],
                                                    constant=setting['data'][im_info['data']]['DefaultPixelValue'])
                else:
                    dvf_ref_sitk = im_sitk
                if im_info['deform_method'] == 'single_frequency':
                    dvf = dvf_generation.single_freq(setting, im_info, stage=stage, im_input_sitk=dvf_ref_sitk)
                elif im_info['deform_method'] == 'mixed_frequency':
                    dvf = dvf_generation.mixed_freq(setting, im_info, stage=stage)
                elif im_info['deform_method'] == 'translation':
                    dvf = dvf_generation.translation(setting, im_info, stage=stage, im_input_sitk=dvf_ref_sitk)
                elif im_info['deform_method'] == 'zero':
                    dvf = dvf_generation.zero(im_input_sitk=dvf_ref_sitk)
                elif im_info['deform_method'] == 'respiratory_motion':
                    dvf = dvf_generation.respiratory_motion(setting, im_info, stage=stage,  moving_image_mode='Exhale')
                else:
                    raise ValueError('DeformMethod= '+im_info['deform_method']+' is not defined. ')

                dvf_sitk = ip.array_to_sitk(dvf, im_ref=dvf_ref_sitk, is_vector=True)
                sitk.WriteImage(sitk.Cast(dvf_sitk, sitk.sitkVectorFloat32), dvf_address)
                if setting['WriteDVFStatistics']:
                    dvf_statistics(setting, dvf, spacing=im_sitk.GetSpacing()[::-1], im_info=im_info, stage=stage)

            deformed_im_clean_address = su.address_generator(setting, 'DeformedIm', deformed_im_ext='Clean', **im_info_su)
            deformed_torso_address = su.address_generator(setting, 'DeformedTorso', **im_info_su)
            deformed_lung_address = su.address_generator(setting, 'DeformedLung', **im_info_su)
            write_intermediate_images = setting['deform_exp'][im_info['deform_exp']]['WriteIntermediateIntensityAugmentation']
            if not os.path.isfile(deformed_im_clean_address) or \
                    (setting['UseTorsoMask'] and not os.path.isfile(deformed_torso_address)) or \
                    (setting['UseLungMask'] and not os.path.isfile(deformed_lung_address)):
                dvf_t = sitk.DisplacementFieldTransform(sitk.Cast(dvf_sitk, sitk.sitkVectorFloat64))  # After this line you cannot save dvf_sitk any more !!!!!!!!!
            if not os.path.isfile(deformed_im_clean_address):
                deformed_im_clean_sitk = ip.resampler_by_transform(im_sitk, dvf_t, im_ref=im_sitk,
                                                                   default_pixel_value=setting['data'][im_info['data']]['DefaultPixelValue'])
                if write_intermediate_images:
                    sitk.WriteImage(sitk.Cast(deformed_im_clean_sitk, setting['data'][im_info['data']]['ImageByte']), deformed_im_clean_address)
            else:
                deformed_im_clean_sitk = sitk.ReadImage(deformed_im_clean_address)
            if setting['UseTorsoMask']:
                deformed_torso_sitk = ip.resampler_by_transform(im_torso_sitk, dvf_t, im_ref=im_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
                sitk.WriteImage(sitk.Cast(deformed_torso_sitk, sitk.sitkInt8), deformed_torso_address)
            if setting['UseLungMask']:
                deformed_lung_sitk = ip.resampler_by_transform(im_lung_sitk, dvf_t, im_ref=im_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
                sitk.WriteImage(sitk.Cast(deformed_lung_sitk, sitk.sitkInt8), deformed_lung_address)

            if deformed_im_clean_address != deformed_im_address:
                deformed_im_ext_combined = ['Clean']
                deformed_im_previous_sitk = sitk.Image(deformed_im_clean_sitk)
                for i_ext, deformed_im_ext_current in enumerate(im_info['deformed_im_ext']):
                    if deformed_im_ext_current != 'Clean':
                        # deformed_im_previous_address = su.address_generator(setting, 'DeformedIm', deformed_im_ext=deformed_im_ext_combined, **im_info_su)
                        # deformed_im_previous_sitk = ip.ReadImage(deformed_im_previous_address, waiting_time=2)
                        deformed_im_ext_combined.append(deformed_im_ext_current)
                        deformed_im_ext_combined_address = su.address_generator(setting, 'DeformedIm', deformed_im_ext=deformed_im_ext_combined, **im_info_su)
                        if not os.path.isfile(deformed_im_ext_combined_address):
                            if deformed_im_ext_current == 'Noise':
                                deformed_im_current_sitk = intensity_augmentation.add_noise(setting, im_info, stage, deformed_im_previous_sitk=deformed_im_previous_sitk)
                            elif deformed_im_ext_current == 'Occluded':
                                deformed_im_current_sitk = intensity_augmentation.add_occlusion(setting, im_info, stage, deformed_im_previous_sitk=deformed_im_previous_sitk)
                            elif deformed_im_ext_current == 'Sponge':
                                deformed_im_current_sitk = intensity_augmentation.add_sponge_model(setting, im_info, stage, deformed_im_previous_sitk=deformed_im_previous_sitk)
                            else:
                                raise ValueError("deformed_im_ext should be in ['Noise', 'Sponge', 'Occluded']")
                            if write_intermediate_images or i_ext == (len(im_info['deformed_im_ext'])-1):
                                sitk.WriteImage(sitk.Cast(deformed_im_current_sitk, setting['data'][im_info['data']]['ImageByte']), deformed_im_ext_combined_address)
                            deformed_im_previous_sitk = sitk.Image(deformed_im_current_sitk)
                        else:
                            deformed_im_previous_sitk = sitk.ReadImage(deformed_im_ext_combined_address)

    dvf_sitk = ip.ReadImage(dvf_address, waiting_time=4)
    dvf = sitk.GetArrayFromImage(dvf_sitk)
    deformed_im_sitk = ip.ReadImage(deformed_im_address, waiting_time=2)
    deformed_im = sitk.GetArrayFromImage(deformed_im_sitk)
    dvf = dvf.astype(np.float32)

    mask_to_zero = setting['deform_exp'][im_info['deform_exp']]['MaskToZero']
    if mask_to_zero is not None:
        im_mask = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, mask_to_zero, **im_info_su)))
        im[im_mask == 0] = setting['data'][im_info['data']]['DefaultPixelValue']
        deformed_mask = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, 'Deformed'+mask_to_zero, **im_info_su)))
        deformed_im[deformed_mask == 0] = setting['data'][im_info['data']]['DefaultPixelValue']

        # UseTorsoMask has the same size as DVF not the ImPad. We've already used im_torso to
        # mask the image. Later we use it only to find the indices. So it should have the
        # the same size as the dvf.
        dvf_pad = setting['DVFPad_S' + str(stage)]
        if dvf_pad > 0:
            im_mask = np.pad(im_mask, dvf_pad, 'constant', constant_values=(0,))
    else:
        im_mask = None

    im_pad = setting['ImPad_S' + str(stage)]
    if im_pad > 0:
        im = np.pad(im, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['DefaultPixelValue'],))
        deformed_im = np.pad(deformed_im, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['DefaultPixelValue'],))

    end_time = time.time()
    if setting['verbose']:
        logging.debug('artifical_generation['+mode_synthetic_dvf+']: Data='+im_info['data']+', TypeIm='+str(im_info['type_im'])+', CN='+str(im_info['cn']) +
                      ' Dsmooth='+str(im_info['dsmooth'])+' D='+str(im_info['deform_number']) +
                      ' is Done in {:.3f}s'.format(end_time - start_time))
    return im, deformed_im, dvf, im_mask


def wait_for_image(im_address, print_address=None):
    if print_address is None:
        print_address = im_address
    count_wait = 1
    while not os.path.isfile(im_address):
        time.sleep(5)
        logging.debug('artifical_generation[reading]: waiting {}s for '.format(count_wait * 5) + print_address)
        count_wait += 1
    return 0


def generate_next_im(setting, im_info, stage):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    next_folder = su.address_generator(setting, 'NextFolder', **im_info_su)
    if not os.path.exists(next_folder):
        os.makedirs(next_folder)
    original_im_sitk = sitk.ReadImage(su.address_generator(setting, 'OriginalIm', **im_info_su))
    next_dvf = dvf_generation.single_freq(setting, im_input_sitk=original_im_sitk, im_info=im_info, stage=stage, gonna_generate_next_im=True)
    next_dvf_sitk = ip.array_to_sitk(next_dvf, is_vector=True, im_ref=original_im_sitk)
    if setting['verbose_image']:
        sitk.WriteImage(sitk.Cast(next_dvf_sitk, sitk.sitkVectorFloat32),
                        su.address_generator(setting, 'NextDVF', **im_info_su))
    dvf_t = sitk.DisplacementFieldTransform(next_dvf_sitk)  # After this line you cannot save NextDVF any more !!!!!!!!!
    next_im_clean_sitk = ip.resampler_by_transform(original_im_sitk, dvf_t, im_ref=original_im_sitk,
                                                   default_pixel_value=setting['data'][im_info['data']]['DefaultPixelValue'])

    if setting['UseLungMask']:
        original_mask_sitk = sitk.ReadImage(su.address_generator(setting, 'OriginalLung', **im_info_su))
        next_mask_sitk = ip.resampler_by_transform(original_mask_sitk, dvf_t, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(sitk.Cast(next_mask_sitk, sitk.sitkInt8),
                        su.address_generator(setting, 'NextLung', **im_info_su))
    if setting['UseTorsoMask']:
        original_torso_sitk = sitk.ReadImage(su.address_generator(setting, 'OriginalTorso', **im_info_su))
        next_torso_sitk = ip.resampler_by_transform(original_torso_sitk, dvf_t, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(sitk.Cast(next_torso_sitk, sitk.sitkInt8),
                        su.address_generator(setting, 'NextTorso', **im_info_su))

    next_im_sponge_sitk = intensity_augmentation.add_sponge_model(setting, im_info, stage=stage,
                                                                  deformed_im_previous_sitk=next_im_clean_sitk,
                                                                  dvf=next_dvf,
                                                                  gonna_generate_next_im=True)
    next_im_sitk = intensity_augmentation.add_noise(setting, im_info, stage=stage,
                                                    deformed_im_previous_sitk=next_im_sponge_sitk,
                                                    gonna_generate_next_im=True)

    sitk.WriteImage(sitk.Cast(next_im_sitk, setting['data'][im_info['data']]['ImageByte']),
                    su.address_generator(setting, 'NextIm', **im_info_su))


def check_downsampled_original_images(setting, im_info, stage):
    padto = im_info['padto']
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth']}

    im_list_downsample = [{'Image': 'OriginalIm',
                           'interpolator': 'BSpline',
                           'DefaultPixelValue': setting['data'][im_info['data']]['DefaultPixelValue'],
                           'ImageByte': setting['data'][im_info['data']]['ImageByte']},
                          {'Image': 'OriginalLung',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8},
                          {'Image': 'OriginalTorso',
                           'interpolator': 'NearestNeighbor',
                           'DefaultPixelValue': 0,
                           'ImageByte': sitk.sitkInt8}
                          ]

    for im_dict in im_list_downsample:
        im_stage_address = su.address_generator(setting, im_dict['Image'], stage=stage, padto=padto, **im_info_su)
        if not os.path.isfile(im_stage_address):
            im_s1_sitk = sitk.ReadImage(su.address_generator(setting, im_dict['Image'], stage=1, padto=None, **im_info_su))
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
                im_stage_sitk = ip.resampler_sitk(im_s1_sitk,
                                                  scale=stage,
                                                  default_pixel_value=im_dict['DefaultPixelValue'],
                                                  interpolator=interpolator)
                # for debugging
                sitk.WriteImage(sitk.Cast(im_stage_sitk, im_dict['ImageByte']),
                                su.address_generator(setting, im_dict['Image'], stage=stage, padto=None, **im_info_su))
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


def check_all_images_exist(setting, im_info, stage, mask_to_zero=None):
    """
    This function check if all images are available and return a boolean.

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
    :param stage

    :return all_exist [boolean]
    """
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto'],
                  'deformed_im_ext': im_info['deformed_im_ext']}

    im_name_list = ['Im', 'DeformedIm', 'DeformedDVF']
    if mask_to_zero is not None:
        im_name_list = im_name_list + [mask_to_zero, 'Deformed'+mask_to_zero]

    all_exist = True
    for im_name in im_name_list:
        im_address = su.address_generator(setting, im_name, **im_info_su)
        if not os.path.isfile(im_address):
            all_exist = False
            break
    return all_exist


def dvf_statistics(setting, dvf, spacing=None, im_info=None, stage=None):
    # input is the dvf in numpy array.
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    max_dvf = np.max(setting['deform_exp'][im_info['deform_exp']]['MaxDeform'])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(np.ravel(dvf), log=True, bins=np.arange(-max_dvf, max_dvf + 1))
    # this range is fine, because in the code the DVF will be normolized to be in range of ()
    plt.draw()
    plt.savefig(su.address_generator(setting, 'DVF_histogram', **im_info_su))
    plt.close()

    jac = ip.calculate_jac(dvf, spacing)
    sitk.WriteImage(sitk.GetImageFromArray(jac.astype(np.float32)), su.address_generator(setting, 'Jac', **im_info_su))
    jac_hist_max = 3
    jac_hist_min = -1
    step_h = 0.2
    if np.max(jac) > jac_hist_max:
        jac_hist_max = np.ceil(np.max(jac))
    if np.min(jac) < jac_hist_min:
        jac_hist_min = np.floor(np.min(jac))

    plt.figure()
    plt.hist(np.ravel(jac), log=True, bins=np.arange(jac_hist_min, jac_hist_max+step_h, step_h))
    plt.title('min(Jac)={:.2f}, max(Jac)={:.2f}'.format(np.min(jac), np.max(jac)))
    plt.draw()
    plt.savefig(su.address_generator(setting, 'Jac_histogram', **im_info_su))
    plt.close()
