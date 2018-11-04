import os
import time
import SimpleITK as sitk
import scipy.ndimage as ndimage
import numpy as np
import logging
from joblib import Parallel, delayed
import multiprocessing
import functions.image_processing as ip
import functions.setting_utils as su
from scipy.ndimage.filters import gaussian_filter


def get_dvf_and_deformed_images(setting, im_info=None, stage=None, mode_synthetic_dvf='generation', stages_to_generate_simultaneously=None, generation_only=False):
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
    :param mode_synthetic_dvf:      'generation', 'reading',  in generationMode only images are generated without reading them.
    :param stages_to_generate_simultaneously:    only used in get_dvf_and_deformed_images_downsampled
    :param generation_only:          only used in get_dvf_and_deformed_images_downsampled

    :return im             image with mentioned ImageType and CN
    :return deformed_im    Deformed image by applying the synthetic DeformedDVF_ on the Im_
    :return dvf            Syntethic DVF
    """
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage}
    # A dictionary of arguments, which only be used in setting_utils.address_generator. This dictionary helps to have shorter code line.

    if stage > 1:
        return get_dvf_and_deformed_images_downsampled(setting, im_info=im_info,
                                                       requested_stage=stage,
                                                       stages_to_generate_simultaneously=stages_to_generate_simultaneously,
                                                       generation_only=generation_only)
    if im_info['dsmooth'] == 0:
        im_sitk = sitk.ReadImage(su.address_generator(setting, 'originalIm', **im_info_su))
        if setting['torsoMask']:
            im_torso_sitk = sitk.ReadImage(su.address_generator(setting, 'originalTorso', **im_info_su))
    else:
        next_im_address = (su.address_generator(setting, 'nextIm', **im_info_su))
        if not os.path.isfile(next_im_address):
            if mode_synthetic_dvf == 'reading':
                count_wait = 1
                while not os.path.isfile(next_im_address):
                    time.sleep(5)
                    logging.debug('SyntheticDeformation[reading]: waiting {}s for '.format(count_wait*5)+su.address_generator(setting, 'nextIm', print_mode=True, **im_info_su))
                    count_wait += 1
            if mode_synthetic_dvf == 'generation':
                logging.debug('SyntheticDeformation[generation]: start '+su.address_generator(setting, 'nextIm', print_mode=True, **im_info_su))
                generate_next_im(setting, im_info=im_info)
        if (time.time() - os.path.getmtime(next_im_address)) < 5:
            time.sleep(5)
            # simpleITK when writing, creates a blank file then fills it within few seconds. This waiting prevents reading blank files.
        im_sitk = sitk.ReadImage(next_im_address)
        if setting['torsoMask']:
            # please note that torso is created a few minutes after nextIm. So we have to wait for the torso.
            im_torso_address = su.address_generator(setting, 'nextTorso', **im_info_su)
            if mode_synthetic_dvf == 'reading':
                if not os.path.isfile(im_torso_address):
                    count_wait = 1
                    while not os.path.isfile(im_torso_address):
                        time.sleep(5)
                        logging.debug('SyntheticDeformation[reading]: waiting {}s for '.format(count_wait*5) +
                                      su.address_generator(setting, 'nextTorso', print_mode=True, **im_info_su))
                        count_wait += 1
            im_torso_sitk = sitk.ReadImage(im_torso_address)

    im = sitk.GetArrayFromImage(im_sitk)
    start_time = time.time()
    dvf_address = su.address_generator(setting, 'deformedDVF', **im_info_su)
    deformed_im_address = su.address_generator(setting, 'deformedIm', **im_info_su)

    if not os.path.isfile(deformed_im_address):
        if mode_synthetic_dvf == 'reading':
            count_wait = 1
            while not os.path.isfile(dvf_address):
                time.sleep(5)
                logging.debug('SyntheticDeformation[reading]: waiting {}s for '.format(count_wait*5) +
                              su.address_generator(setting, 'deformedDVF', print_mode=True, **im_info_su))
                count_wait += 1

            count_wait = 1
            while not os.path.isfile(deformed_im_address):
                time.sleep(5)
                logging.debug('SyntheticDeformation[reading]: waiting {}s for '.format(count_wait*5) +
                              su.address_generator(setting, 'deformedIm', print_mode=True, **im_info_su))
                count_wait += 1
        if mode_synthetic_dvf == 'generation':
            logging.debug('SyntheticDeformation[generation]: start ' +
                          su.address_generator(setting, 'deformedDVF', print_mode=True, **im_info_su))
            if not os.path.exists(su.address_generator(setting, 'DFolder', **im_info_su)):
                os.makedirs(su.address_generator(setting, 'DFolder', **im_info_su))

            if setting['DVFPad_S1'] > 0:
                pad = setting['DVFPad_S1']
                im_input_sitk = sitk.ConstantPad(im_sitk, [pad, pad, pad], [pad, pad, pad],
                                                 constant=setting['data'][im_info['data']]['defaultPixelValue'])
            else:
                im_input_sitk = im_sitk
            if im_info['deform_method'] == 'smoothBSpline':
                dvf = smooth_bspline(setting, im_input_sitk=im_input_sitk, im_info=im_info)
            elif im_info['deform_method'] == 'dilatedEdgeSmooth':
                dvf = dilated_edge_smooth(setting, im_info=im_info)
            elif im_info['deform_method'] == 'translation':
                dvf = translation(setting, im_input_sitk, im_info=im_info)
            else:
                raise ValueError('DeformMethod= '+im_info['deform_method']+' is not defined. ')

            dvf_sitk = ip.array_to_sitk(dvf, im_ref=im_input_sitk, is_vector=True)
            sitk.WriteImage(sitk.Cast(dvf_sitk, sitk.sitkVectorFloat32), dvf_address)
            dvf_t = sitk.DisplacementFieldTransform(dvf_sitk)    # After this line you cannot save dvf_sitk any more !!!!!!!!!
            deformed_im_clean_sitk = ip.resampler_by_dvf(im_sitk, dvf_t, im_ref=im_sitk,
                                                         default_pixel_value=setting['data'][im_info['data']]['defaultPixelValue'])
            # This is the clean version of the deformed image. Intensity noise should be added to this image
            deformed_im_sitk = sitk.AdditiveGaussianNoise(deformed_im_clean_sitk, setting['deform_exp'][im_info['deform_exp']]['sigmaN'], 0, 0)
            sitk.WriteImage(sitk.Cast(deformed_im_sitk, setting['data'][im_info['data']]['imageByte']), deformed_im_address)
            if setting['WriteDVFStatistics']:
                dvf_statistics(setting, dvf, spacing=im_sitk.GetSpacing()[::-1], im_info=im_info, stage=stage)
            if setting['torsoMask']:
                deformed_torso_sitk = ip.resampler_by_dvf(im_torso_sitk, dvf_t, im_ref=im_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
                sitk.WriteImage(sitk.Cast(deformed_torso_sitk, sitk.sitkInt8),
                                su.address_generator(setting, 'deformedTorso', **im_info_su))

    if (time.time() - os.path.getmtime(dvf_address)) < 5:
        time.sleep(5)
        # simpleITK when writing, creates a blank file then fills it within few seconds. This waiting prevents reading blank files.
    dvf_sitk = sitk.ReadImage(dvf_address)
    dvf = sitk.GetArrayFromImage(dvf_sitk)
    if (time.time() - os.path.getmtime(deformed_im_address)) < 5:
        time.sleep(5)
        # simpleITK when writing, creates a blank file then fills it within few seconds. This waiting prevents reading blank files.
    deformed_im_sitk = sitk.ReadImage(deformed_im_address)
    deformed_im = sitk.GetArrayFromImage(deformed_im_sitk)
    dvf = dvf.astype(np.float32)

    if setting['torsoMask']:
        im_torso = (sitk.GetArrayFromImage(im_torso_sitk)).astype(np.bool)
        im[im_torso == 0] = setting['data'][im_info['data']]['defaultPixelValue']
        if not os.path.isfile(su.address_generator(setting, 'deformedTorso', **im_info_su)):
            dvf_t = sitk.DisplacementFieldTransform(sitk.Cast(dvf_sitk, sitk.sitkVectorFloat64))
            deformed_torso_sitk = ip.resampler_by_dvf(im_torso_sitk, dvf_t, im_ref=im_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
            sitk.WriteImage(sitk.Cast(deformed_torso_sitk, sitk.sitkInt8), su.address_generator(setting, 'deformedTorso', **im_info_su))
            time.sleep(5)
        deformed_torso = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, 'deformedTorso', **im_info_su)))
        deformed_im[deformed_torso == 0] = setting['data'][im_info['data']]['defaultPixelValue']
    else:
        im_torso = None

    im_pad = setting['ImPad_S' + str(stage)]
    if im_pad > 0:
        im = np.pad(im, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['defaultPixelValue'],))
        deformed_im = np.pad(deformed_im, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['defaultPixelValue'],))

    dvf_pad = setting['DVFPad_S' + str(stage)]
    if dvf_pad > 0:
        if setting['torsoMask']:
            # torsoMask has the same size as DVF not the ImPad. We've already used im_torso to
            # mask the image. Later we use it onlly to find the indices. So it should have the
            # the same size as the dvf.
            im_torso = np.pad(im_torso, dvf_pad, 'constant', constant_values=(0,))

    end_time = time.time()
    if setting['verbose']:
        logging.debug('SyntheticDeformation['+mode_synthetic_dvf+']: Data='+im_info['data']+', TypeIm='+str(im_info['type_im'])+', CN='+str(im_info['cn']) +
                      ' Dsmooth='+str(im_info['dsmooth'])+' D='+str(im_info['deform_number']) +
                      ' is Done in {:.3f}s'.format(end_time - start_time))
    return im, deformed_im, dvf, im_torso


def smooth_bspline(setting, im_input_sitk=None, im_info=None, gonna_generate_next_im=False):
    seed_number = sum([ord(i) for i in im_info['data']])*1000 + im_info['type_im']*1000 + im_info['cn']*500 + im_info['dsmooth']*20
    deform_number = im_info['deform_number']
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': 1}
    if gonna_generate_next_im:
        max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform_nextIm']
        dim_im = '3D'          # The deformation of the nextIm is always 3D
        seed_number = seed_number + 1
        grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['setGridBorderToZero'][0]
        grid_spacing = setting['deform_exp'][im_info['deform_exp']]['BsplineGridSpacing_smooth'][0]
        grid_smoothing_sigma = setting['deform_exp'][im_info['deform_exp']]['GridSmoothingSigma'][0]
        bspline_transform_address = su.address_generator(setting, 'nextBSplineTransform', **im_info_su)
        bspline_im_address = su.address_generator(setting, 'nextBSplineTransformIm', **im_info_su)
    else:
        max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'][deform_number]
        dim_im = setting['Dim']
        grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['setGridBorderToZero'][deform_number]
        grid_spacing = setting['deform_exp'][im_info['deform_exp']]['BsplineGridSpacing_smooth'][deform_number]
        grid_smoothing_sigma = setting['deform_exp'][im_info['deform_exp']]['GridSmoothingSigma'][deform_number]
        bspline_transform_address = su.address_generator(setting, 'BSplineTransform', **im_info_su)
        bspline_im_address = su.address_generator(setting, 'BSplineTransformIm', **im_info_su)
    np.random.seed(seed_number)

    if setting['DVFPad_S1'] > 0:
        # im_input is already zeropadded in this case
        padded_mm = setting['DVFPad_S1'] * im_input_sitk.GetSpacing()[0]
        grid_border_to_zero = (grid_border_to_zero + np.ceil(np.repeat(padded_mm, int(dim_im[0])) / grid_spacing)).astype(np.int)
        if len(np.unique(im_input_sitk.GetSpacing())) > 1:
            raise ValueError('SyntheticDeformation: padding is only implemented for isotropic voxel size. current voxel size = [{}, {}, {}]'.format(
                im_input_sitk.GetSpacing()[0], im_input_sitk.GetSpacing()[1], im_input_sitk.GetSpacing()[2]))

    number_of_grids = list(np.round(np.array(im_input_sitk.GetSize()) * np.array(im_input_sitk.GetSpacing()) / grid_spacing))
    number_of_grids = [int(i) for i in number_of_grids]  # This is a bit funny, it has to be int (and not even np.int)
    # BCoeff = sitk.BSplineTransformInitializer(ImInput, numberOfGrids, order=3)
    # problem with the offset
    bcoeff = sitk.BSplineTransformInitializer(sitk.Image(im_input_sitk.GetSize(), sitk.sitkInt8), number_of_grids, order=3)
    bcoeff_parameters = np.random.uniform(-max_deform*4, max_deform*4, len(bcoeff.GetParameters()))
    # we choose numbers to be in range of MaxDeform, please note that there are two smoothing steps after this initialization.
    # So numbers will be much smaller.

    grid_side = bcoeff.GetTransformDomainMeshSize()
    if dim_im == '3D':
        bcoeff_smoothed_dim = [None] * 3
        for dim in range(3):
            bcoeff_dim = np.reshape(np.split(bcoeff_parameters, 3)[dim], [grid_side[2]+3, grid_side[1]+3, grid_side[0]+3])
            # number of coefficients in grid is increased with 3 in simpleITK.
            if np.any(grid_border_to_zero):
                # in two steps, the marginal coefficient of the grids are set to zero:
                # 1. before smoothing the grid with gridBorderToZero+1  2. after smoothing the grid with gridBorderToZero
                non_zero_mask = np.zeros(np.shape(bcoeff_dim))
                non_zero_mask[grid_border_to_zero[0] + 1:-grid_border_to_zero[0] - 1, grid_border_to_zero[1] + 1:-grid_border_to_zero[1] - 1,
                              grid_border_to_zero[2] + 1:-grid_border_to_zero[2] - 1] = 1
                bcoeff_dim = bcoeff_dim * non_zero_mask
            bcoeff_smoothed_dim[dim] = ndimage.filters.gaussian_filter(bcoeff_dim, grid_smoothing_sigma[dim])
            if np.any(grid_border_to_zero):
                non_zero_mask = np.zeros(np.shape(bcoeff_dim))
                non_zero_mask[grid_border_to_zero[0]:-grid_border_to_zero[0], grid_border_to_zero[1]:-grid_border_to_zero[1],
                              grid_border_to_zero[2]:-grid_border_to_zero[2]] = 1
                bcoeff_smoothed_dim[dim] = bcoeff_smoothed_dim[dim] * non_zero_mask

        bcoeff_parameters_smooth = np.hstack((np.reshape(bcoeff_smoothed_dim[0], -1),
                                              np.reshape(bcoeff_smoothed_dim[1], -1),
                                              np.reshape(bcoeff_smoothed_dim[2], -1)))
    bcoeff_parameters_smooth_normalize = normalize_dvf(bcoeff_parameters_smooth, max_deform * 1.7)
    bcoeff.SetParameters(bcoeff_parameters_smooth_normalize)

    if setting['WriteBSplineTransform']:
        sitk.WriteTransform(bcoeff, bspline_transform_address)
        bspline_im_sitk_tuple = bcoeff.GetCoefficientImages()
        bspline_im = np.concatenate((np.expand_dims(sitk.GetArrayFromImage(bspline_im_sitk_tuple[0]), axis=-1),
                                     np.expand_dims(sitk.GetArrayFromImage(bspline_im_sitk_tuple[1]), axis=-1),
                                     np.expand_dims(sitk.GetArrayFromImage(bspline_im_sitk_tuple[1]), axis=-1)),
                                    axis=-1)
        bspline_spacing = bspline_im_sitk_tuple[0].GetSpacing()
        bspling_origin = [list(bspline_im_sitk_tuple[0].GetOrigin())[i] + list(im_input_sitk.GetOrigin())[i] for i in range(3)]
        bspline_direction = im_input_sitk.GetDirection()
        bspline_im_sitk = ip.array_to_sitk(bspline_im, origin=bspling_origin, spacing=bspline_spacing, direction=bspline_direction, is_vector=True)
        sitk.WriteImage(bspline_im_sitk, bspline_im_address)

    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_input_sitk.GetSize())
    dvf_sitk = dvf_filter.Execute(bcoeff)
    dvf = sitk.GetArrayFromImage(dvf_sitk)
    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform)
    return dvf


def translation(setting, im_input_sitk, im_info=None):
    seed_number = sum([ord(i) for i in im_info['data']])*1000 + im_info['type_im']*1100 + im_info['cn']*510 + im_info['dsmooth']*21
    deform_number = im_info['deform_number']
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'][deform_number]
    dim_im = setting['Dim']
    grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['setGridBorderToZero_translation'][deform_number]
    grid_spacing = setting['deform_exp'][im_info['deform_exp']]['BsplineGridSpacing_translation'][deform_number]
    np.random.seed(seed_number)

    if setting['DVFPad_S1'] > 0:
        # ImInput is already zeropadded in this case
        padded_mm = setting['DVFPad_S1'] * im_input_sitk.GetSpacing()[0]
        grid_border_to_zero = (grid_border_to_zero + np.ceil(np.repeat(padded_mm, int(dim_im[0])) / grid_spacing)).astype(np.int)
        if len(np.unique(im_input_sitk.GetSpacing())) > 1:
            raise ValueError('SyntheticDeformation: padding is only implemented for isotropic voxel size. current voxel size = [{}, {}, {}]'.format(
                im_input_sitk.GetSpacing()[0], im_input_sitk.GetSpacing()[1], im_input_sitk.GetSpacing()[2]))

    number_of_grids = list(np.round(np.array(im_input_sitk.GetSize()) * np.array(im_input_sitk.GetSpacing()) / grid_spacing))
    number_of_grids = [int(i) for i in number_of_grids]  # it has to be int (and not even np.int)
    # BCoeff = sitk.BSplineTransformInitializer(ImInput, numberOfGrids, order=3)
    # problem with the offset
    bcoeff = sitk.BSplineTransformInitializer(sitk.Image(im_input_sitk.GetSize(), sitk.sitkInt8), number_of_grids, order=3)
    grid_side = bcoeff.GetTransformDomainMeshSize()
    if dim_im == '3D':
        bcoeff_smoothed_dim = [None] * 3
        translation_magnitude = [None] * 3
        for dim in range(3):
            if np.random.ranf() > 0.8:
                translation_magnitude[dim] = 0
            else:
                translation_magnitude[dim] = np.random.uniform(-max_deform, max_deform)
            if dim == 2:
                if translation_magnitude[2] < max_deform * 2/3:
                    if translation_magnitude[1] < max_deform * 2/3:
                        if translation_magnitude[0] < max_deform * 2/3:
                            translation_magnitude[2] = np.random.uniform(max_deform * 2/3, max_deform)
                            sign_of_magnitude = np.random.ranf()
                            if sign_of_magnitude > 0.5:
                                translation_magnitude[2] = -  translation_magnitude[2]

            bcoeff_dim = np.ones([grid_side[2]+3, grid_side[1]+3, grid_side[0]+3]) * translation_magnitude[dim]
            # number of coefficients in grid is increased with 3 in simpleITK.
            if np.any(grid_border_to_zero):
                non_zero_mask = np.zeros(np.shape(bcoeff_dim))
                non_zero_mask[grid_border_to_zero[0]:-grid_border_to_zero[0], grid_border_to_zero[1]:-grid_border_to_zero[1],
                              grid_border_to_zero[2]:-grid_border_to_zero[2]] = 1
                bcoeff_dim = bcoeff_dim * non_zero_mask
            bcoeff_smoothed_dim[dim] = bcoeff_dim
        bcoeff_parameters_smooth = np.hstack((np.reshape(bcoeff_smoothed_dim[0], -1),
                                   np.reshape(bcoeff_smoothed_dim[1], -1),
                                   np.reshape(bcoeff_smoothed_dim[2], -1)))

    bcoeff.SetParameters(bcoeff_parameters_smooth)
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_input_sitk.GetSize())
    dvf_sitk = dvf_filter.Execute(bcoeff)
    dvf = sitk.GetArrayFromImage(dvf_sitk)
    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform)
    return dvf


def dilated_edge_smooth(setting, im_info=None, stage=1):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage}
    seed_number = sum([ord(i) for i in im_info['data']]) * 1050 + im_info['type_im'] * 950 + im_info['cn'] * 480 + im_info['dsmooth'] * 13
    np.random.seed(seed_number)
    deform_number = im_info['deform_number']
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'][deform_number]
    grid_smoothing_sigma = setting['deform_exp'][im_info['deform_exp']]['GridSmoothingSigma_dilatedEdge'][deform_number]
    grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['setGridBorderToZero_dilatedEdge'][deform_number]
    grid_spacing = setting['deform_exp'][im_info['deform_exp']]['BsplineGridSpacing_dilatedEdge'][deform_number]  # Approximately
    number_dilation = setting['deform_exp'][im_info['deform_exp']]['Np_dilateEdge'][deform_number]

    im_canny_address = su.address_generator(setting, 'ImCanny', **im_info_su)
    im_sitk = sitk.ReadImage(su.address_generator(setting, 'Im', **im_info_su))
    if os.path.isfile(im_canny_address):
        im_canny_sitk = sitk.ReadImage(im_canny_address)
    else:
        im_canny_sitk = sitk.CannyEdgeDetection(sitk.Cast(im_sitk, sitk.sitkFloat32),
                                                lowerThreshold=setting['deform_exp'][im_info['deform_exp']]['onEdge-lowerThreshold'],
                                                upperThreshold=setting['deform_exp'][im_info['deform_exp']]['onEdge-upperThreshold'])
        sitk.WriteImage(sitk.Cast(im_canny_sitk, sitk.sitkInt8), im_canny_address)
    mask_ = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, 'Mask', **im_info_su))).astype(np.bool)
    im_canny = sitk.GetArrayFromImage(im_canny_sitk)
    # erosion with ndimage is 5 times faster than SimpleITK
    mask_dilated = ndimage.binary_dilation(mask_)
    available_region = np.logical_and(mask_dilated, im_canny)
    available_region = np.tile(np.expand_dims(available_region, axis=-1), 3)
    dilated_edge = np.copy(available_region)

    itr_edge = 0
    i_edge = [None]*3
    select_voxel = [None]*3
    block_low = [None]*3
    block_high = [None]*3
    for dim in range(3):
        i_edge[dim] = np.where(available_region[:, :, :, dim] > 0)
        # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
    if (len(i_edge[0][0]) == 0) or (len(i_edge[1][0]) == 0) or (len(i_edge[2][0]) == 0):
        logging.debug('SyntheticDeformation: We are out of points. Plz change the threshold value of Canny method!!!!! ')  # Old method. only edges!
    while (len(i_edge[0][0]) > 4) and (len(i_edge[1][0]) > 4) and (len(i_edge[2][0]) > 4) and (itr_edge < number_dilation):
        # i_edge will change at the end of this while loop!
        no_more_dilatation_in_this_region = False
        for dim in range(3):
            select_voxel[dim] = int(np.random.randint(0, len(i_edge[dim][0]) - 1, 1, dtype=np.int64))
            block_low[dim], block_high[dim] = center_to_block(setting,
                                                              center=np.array([i_edge[dim][0][select_voxel[dim]],
                                                                               i_edge[dim][1][select_voxel[dim]],
                                                                               i_edge[dim][2][select_voxel[dim]]]),
                                                              radius=setting['deform_exp'][im_info['deform_exp']]['blockRadius_dilatedEdge'],
                                                              im_ref=im_sitk)
        if itr_edge == 0:
            struct = np.ones((3, 3, 3), dtype=bool)
            for dim in range(3):
                dilated_edge[:, :, :, dim] = ndimage.binary_dilation(dilated_edge[:, :, :, dim], structure=struct)

        elif itr_edge < np.round(10*number_dilation/12):  # We like to include zero deformation in our training set.
            no_more_dilatation_in_this_region = True
            for dim in range(3):
                dilated_edge[block_low[dim][0]:block_high[dim][0],
                             block_low[dim][1]:block_high[dim][1],
                             block_low[dim][2]:block_high[dim][2], dim] = False

        elif itr_edge < np.round(11*number_dilation/12):
            struct = ndimage.generate_binary_structure(3, 2)
            for dim in range(3):
                mask_for_edge_dilation = np.zeros(np.shape(dilated_edge[:, :, :, dim]), dtype=bool)
                mask_for_edge_dilation[block_low[dim][0]:block_high[dim][0], block_low[dim][1]:block_high[dim][1], block_low[dim][2]:block_high[dim][2]] = True
                dilated_edge[:, :, :, dim] = ndimage.binary_dilation(dilated_edge[:, :, :, dim], structure=struct, mask=mask_for_edge_dilation)
            if (itr_edge % 2) == 0:
                no_more_dilatation_in_this_region = True
        elif itr_edge < number_dilation:
            struct = np.zeros((9, 9, 9), dtype=bool)
            if (itr_edge % 3) == 0:
                struct[0:5, :, :] = True
            if (itr_edge % 3) == 1:
                struct[:, 0:5, :] = True
            if (itr_edge % 3) == 2:
                struct[:, :, 0:5] = True
            for dim in range(3):
                mask_for_edge_dilation = np.zeros(np.shape(dilated_edge[:, :, :, dim]), dtype=bool)
                mask_for_edge_dilation[block_low[dim][0]:block_high[dim][0], block_low[dim][1]:block_high[dim][1], block_low[dim][2]:block_high[dim][2]] = True
                dilated_edge[:, :, :, dim] = ndimage.binary_dilation(dilated_edge[:, :, :, dim], structure=struct, mask=mask_for_edge_dilation)
            if np.random.uniform() > 0.3:
                no_more_dilatation_in_this_region = True
        if no_more_dilatation_in_this_region:
            available_region[block_low[dim][0]:block_high[dim][0], block_low[dim][1]:block_high[dim][1], block_low[dim][2]:block_high[dim][2], dim] = False
        if itr_edge >= np.round(10*number_dilation/12):
            for dim in range(3):
                i_edge[dim] = np.where(available_region[:, :, :, dim] > 0)
        itr_edge += 1
        # logging.debug('SyntheticDeformation: dilatedEdge itr_edge={}'.format(itr_edge))

    number_of_grids = list(np.round(np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing()) / grid_spacing))
    number_of_grids = [int(i) for i in number_of_grids]  # it has to be int (and not even np.int)
    bcoeff = sitk.BSplineTransformInitializer(sitk.Image(im_sitk.GetSize(), sitk.sitkInt8), number_of_grids, order=3)
    bcoeff_parameters = np.random.uniform(-max_deform*4, max_deform*4, len(bcoeff.GetParameters()))
    grid_side = bcoeff.GetTransformDomainMeshSize()

    bcoeff_smoothed_dim = [None] * 3
    for dim in range(3):
        bcoeff_dim = np.reshape(np.split(bcoeff_parameters, 3)[dim], [grid_side[2]+3, grid_side[1]+3, grid_side[0]+3])
        # number of coefficients in grid is increased with 3 in simpleITK.
        if np.any(grid_border_to_zero):
            non_zero_mask = np.zeros(np.shape(bcoeff_dim))
            non_zero_mask[grid_border_to_zero[0]:-grid_border_to_zero[0], grid_border_to_zero[1]:-grid_border_to_zero[1],
                          grid_border_to_zero[2]:-grid_border_to_zero[2]] = 1
            bcoeff_dim = bcoeff_dim * non_zero_mask
        bcoeff_smoothed_dim[dim] = ndimage.filters.gaussian_filter(bcoeff_dim, grid_smoothing_sigma[dim])
    bcoeff_parameters_smooth = np.hstack((np.reshape(bcoeff_smoothed_dim[0], -1),
                                          np.reshape(bcoeff_smoothed_dim[1], -1),
                                          np.reshape(bcoeff_smoothed_dim[2], -1)))
    bcoeff_parameters_smooth_normalize = normalize_dvf(bcoeff_parameters_smooth, max_deform * 2)
    bcoeff.SetParameters(bcoeff_parameters_smooth_normalize)
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_sitk.GetSize())
    smoothed_values_sitk = dvf_filter.Execute(bcoeff)
    smoothed_values = sitk.GetArrayFromImage(smoothed_values_sitk)

    dvf = (dilated_edge.astype(np.float64) * smoothed_values).astype(np.float64)
    if setting['DVFPad_S1'] > 0:
        pad = setting['DVFPad_S1']
        dvf = np.pad(dvf, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0,))

    sigma = np.random.uniform(low=setting['deform_exp'][im_info['deform_exp']]['sigmaRange_dilatedEdge'][deform_number][0],
                              high=setting['deform_exp'][im_info['deform_exp']]['sigmaRange_dilatedEdge'][deform_number][1],
                              size=3)
    dvf = smooth_dvf(dvf, sigma_blur=sigma, parallel_processing=setting['ParallelProcessing'])

    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, setting['deform_exp'][im_info['deform_exp']]['MaxDeform'][deform_number])

    # sitk.WriteImage(sitk.GetImageFromArray(dilated_Edge.astype(np.int8), isVector=True), DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/' + 'dilated_Edge.mha')
    # sitk.WriteImage(sitk.GetImageFromArray(availableRegion.astype(np.int8), isVector=True), DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/' + 'availableRegion.mha')
    return dvf


def smooth_dvf(dvf, dim_im='3D', sigma_blur=None, parallel_processing=True):
    dvf_smooth = np.empty(np.shape(dvf))
    if parallel_processing:
        num_cores = multiprocessing.cpu_count() - 2
        if dim_im == '3D':
            # The following line is not working in Windows
            [dvf_smooth[:, :, :, 0], dvf_smooth[:, :, :, 1], dvf_smooth[:, :, :, 2]] = \
                Parallel(n_jobs=num_cores)(delayed(smooth_gaussian)(dvf=dvf[:, :, :, i], sigma=sigma_blur[i]) for i in range(np.shape(dvf)[3]))
        if dim_im == '2D':
            [dvf_smooth[:, :, :, 0], dvf_smooth[:, :, :, 1]] = \
                Parallel(n_jobs=num_cores)(delayed(smooth_gaussian)(dvf=dvf[:, :, :, i], sigma=sigma_blur[i]) for i in range(np.shape(dvf)[3]))
            dvf_smooth[:, :, :, 2] = dvf[:, :, :, 2]
    else:
        if dim_im == '3D':
            for dim in range(3):
                dvf_smooth[:, :, :, dim] = smooth_gaussian(dvf[:, :, :, dim], sigma_blur[dim])
    return dvf_smooth


def generate_next_im(setting, im_info=None, stage=1):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage}
    next_folder = su.address_generator(setting, 'nextFolder', **im_info_su)
    if not os.path.exists(next_folder):
        os.makedirs(next_folder)
    original_im_sitk = sitk.ReadImage(su.address_generator(setting, 'originalIm', **im_info_su))
    next_dvf = smooth_bspline(setting, im_input_sitk=original_im_sitk, im_info=im_info, gonna_generate_next_im=True)
    next_dvf_sitk = ip.array_to_sitk(next_dvf, is_vector=True, im_ref=original_im_sitk)
    if setting['verbose_image']:
        sitk.WriteImage(sitk.Cast(next_dvf_sitk, sitk.sitkVectorFloat32),
                        su.address_generator(setting, 'nextDVF', **im_info_su))
    dvf_t = sitk.DisplacementFieldTransform(next_dvf_sitk)  # After this line you cannot save nextDVF any more !!!!!!!!!
    next_im_clean_sitk = ip.resampler_by_dvf(original_im_sitk, dvf_t, im_ref=original_im_sitk,
                                             default_pixel_value=setting['data'][im_info['data']]['defaultPixelValue'])
    next_im_sitk = sitk.AdditiveGaussianNoise(next_im_clean_sitk,
                                              setting['deform_exp'][im_info['deform_exp']]['sigmaN_nextIm'],
                                              0, 0)
    sitk.WriteImage(sitk.Cast(next_im_sitk, setting['data'][im_info['data']]['imageByte']),
                    su.address_generator(setting, 'nextIm', **im_info_su))
    if setting['loadMask']:
        original_mask_sitk = sitk.ReadImage(su.address_generator(setting, 'originalMask', **im_info_su))
        next_mask_sitk = ip.resampler_by_dvf(original_mask_sitk, dvf_t, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(sitk.Cast(next_mask_sitk, sitk.sitkInt8),
                        su.address_generator(setting, 'nextMask', **im_info_su))
    if setting['torsoMask']:
        original_torso_sitk = sitk.ReadImage(su.address_generator(setting, 'originalTorso', **im_info_su))
        next_torso_sitk = ip.resampler_by_dvf(original_torso_sitk, dvf_t, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(sitk.Cast(next_torso_sitk, sitk.sitkInt8),
                        su.address_generator(setting, 'nextTorso', **im_info_su))


def center_to_block(setting, center=None, radius=10, im_ref=None):
    block_low = center - radius
    block_high = center + radius
    if setting['Dim'] == '2D':
        block_low[0] = center[0] - 1
        block_high[0] = center[0] + 2
    for dim in range(3):
        if block_low[dim] < 0:
            block_low[dim] = 0
        if block_high[dim] > im_ref.GetSize()[-1-dim]:
            block_high[dim] = im_ref.GetSize()[-1-dim]
    return block_low, block_high


def dvf_statistics(setting, dvf, spacing=None, im_info=None, stage=None):
    # input is the dvf in numpy array.
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage}
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


def normalize_dvf(dvf, max_deform):
    max_value = max(abs(np.max(dvf)), abs(np.min(dvf)))
    if max_value > max_deform:
        dvf = dvf * max_deform / max_value
    return dvf


def smooth_gaussian(dvf, sigma):
    return gaussian_filter(dvf, sigma=sigma)


def get_dvf_and_deformed_images_downsampled(setting=None, im_info=None, requested_stage=None, stages_to_generate_simultaneously=None, generation_only=False):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth']}
    if stages_to_generate_simultaneously is None:
        logging.debug('stages_to_generate_simultaneously will set to ' + str(requested_stage))
        stages_to_generate_simultaneously = np.array([requested_stage], dtype=np.int)

    im_stage_address = su.address_generator(setting, 'Im', stage=requested_stage, **im_info_su)
    deformed_im_stage_address = su.address_generator(setting, 'deformedIm', stage=requested_stage, **im_info_su)
    dvf_stage_address = su.address_generator(setting, 'deformedDVF', stage=requested_stage, **im_info_su)

    all_files_exist = os.path.isfile(im_stage_address) and os.path.isfile(deformed_im_stage_address) and os.path.isfile(dvf_stage_address)
    if setting['torsoMask']:
        torso_stage_address = su.address_generator(setting, 'Torso', stage=requested_stage, **im_info_su)
        deformed_torso_stage_address = su.address_generator(setting, 'deformedTorso', stage=requested_stage, **im_info_su)
        all_files_exist = all_files_exist and os.path.isfile(torso_stage_address) and os.path.isfile(deformed_torso_stage_address)

    if not all_files_exist:
        logging.debug('SyntheticDeformation: downsamlped images not found data='+im_info['data'] +
                      ' TypeIm={}, CN={}, Dsmooth={}. Start generating...... I need one free GPU for downsampling'.format(im_info['type_im'], im_info['cn'], im_info['dsmooth']))
        im_address = su.address_generator(setting, 'Im', stage=1, **im_info_su)
        deformed_im_address = su.address_generator(setting, 'deformedIm', stage=1, **im_info_su)
        dvf_address = su.address_generator(setting, 'deformedDVF', stage=1, **im_info_su)
        all_stage0_exist = os.path.isfile(im_address) and os.path.isfile(deformed_im_address) and os.path.isfile(dvf_address)
        if setting['torsoMask']:
            torso_address = su.address_generator(setting, 'Torso', stage=1, **im_info_su)
            deformed_torso_address = su.address_generator(setting, 'deformedTorso', stage=1, **im_info_su)
            all_stage0_exist = all_stage0_exist and os.path.isfile(torso_address) and os.path.isfile(deformed_torso_address)
        if not all_stage0_exist:
            get_dvf_and_deformed_images(setting, im_info=im_info, stage=1, mode_synthetic_dvf='generation')

        im_sitk = sitk.ReadImage(im_address)
        im = sitk.GetArrayFromImage(im_sitk)
        deformed_im_sitk = sitk.ReadImage(deformed_im_address)
        deformed_im = sitk.GetArrayFromImage(deformed_im_sitk)
        dvf_sitk = sitk.ReadImage(dvf_address)
        dvf = sitk.GetArrayFromImage(dvf_sitk)

        if setting['torsoMask']:
            torso_sitk = sitk.ReadImage(torso_address)
            deformed_torso_sitk = sitk.ReadImage(deformed_torso_address)

        for stage in stages_to_generate_simultaneously:
            time_before = time.time()
            im_stage_address = su.address_generator(setting, 'Im', stage=stage, **im_info_su)
            deformed_im_stage_address = su.address_generator(setting, 'deformedIm', stage=stage, **im_info_su)
            dvf_stage_address = su.address_generator(setting, 'deformedDVF', stage=stage, **im_info_su)
            torso_stage_address = su.address_generator(setting, 'Torso', stage=stage, **im_info_su)
            deformed_torso_stage_address = su.address_generator(setting, 'deformedTorso', stage=stage, **im_info_su)

            im_stage = ip.downsampler_gpu(im, stage, normalizeKernel=True,
                                          default_pixel_value=setting['data'][im_info['data']]['defaultPixelValue'])
            deformed_im_stage = ip.downsampler_gpu(deformed_im, stage, normalizeKernel=True,
                                                   default_pixel_value=setting['data'][im_info['data']]['defaultPixelValue'])
            dvf_stage = [None]*3

            if setting['DVFPad_S1'] % stage:
                extra_pad = setting['DVFPad_S1'] % stage
                dvf_pad = np.pad(dvf, ((extra_pad, extra_pad), (extra_pad, extra_pad), (extra_pad, extra_pad), (0, 0)), 'constant', constant_values=(0,))
            else:
                extra_pad = 0
                dvf_pad = dvf
            pad_of_dvf = setting['DVFPad_S1'] + extra_pad

            for dim in range(np.shape(dvf_pad)[3]):
                dvf_stage[dim] = ip.downsampler_gpu(dvf_pad[:, :, :, dim], stage, normalizeKernel=True, default_pixel_value=0)
            dvf_stage = np.stack([dvf_stage[i] for i in range(3)], axis=3)

            extra_pad_to_do = int(setting['DVFPad_S'+str(stage)] - pad_of_dvf/stage)
            dvf_stage = np.pad(dvf_stage, ((extra_pad_to_do, extra_pad_to_do),
                                           (extra_pad_to_do, extra_pad_to_do),
                                           (extra_pad_to_do, extra_pad_to_do)
                                           , (0, 0)),
                               'constant', constant_values=(0,))
            total_pad = setting['DVFPad_S1'] + extra_pad + extra_pad_to_do * stage

            dvf_stage_sitk = ip.array_to_sitk(dvf_stage,
                                              origin=tuple([im_sitk.GetOrigin()[i] - total_pad for i in range(3)]),
                                              spacing=tuple(i * stage for i in im_sitk.GetSpacing()),
                                              direction=im_sitk.GetDirection(),
                                              is_vector=True)
            im_stage_sitk = ip.array_to_sitk(im_stage, origin=im_sitk.GetOrigin(), spacing=tuple(i * stage for i in im_sitk.GetSpacing()), direction=im_sitk.GetDirection())
            deformed_im_stage_sitk = ip.array_to_sitk(deformed_im_stage, origin=deformed_im_sitk.GetOrigin(),
                                                      spacing=tuple(i*stage for i in deformed_im_sitk.GetSpacing()), direction=deformed_im_sitk.GetDirection())

            sitk.WriteImage(sitk.Cast(im_stage_sitk, setting['data'][im_info['data']]['imageByte']), im_stage_address)
            sitk.WriteImage(sitk.Cast(deformed_im_stage_sitk, setting['data'][im_info['data']]['imageByte']), deformed_im_stage_address)
            sitk.WriteImage(sitk.Cast(dvf_stage_sitk, sitk.sitkVectorFloat32), dvf_stage_address)
            if setting['torsoMask']:
                torso_stage_sitk = ip.downsampler_sitk(torso_sitk, stage, im_ref=im_stage_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
                deformed_torso_stage_sitk = ip.downsampler_sitk(deformed_torso_sitk, stage, im_ref=im_stage_sitk, default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
                sitk.WriteImage(sitk.Cast(torso_stage_sitk, sitk.sitkInt8), torso_stage_address)
                sitk.WriteImage(sitk.Cast(deformed_torso_stage_sitk, sitk.sitkInt8), deformed_torso_stage_address)

            time_after = time.time()
            logging.debug('downsampling data='+im_info['data']+' stage={}, TypeIm={}, CN={}, DSmooth={}  done in {:.2f} s.'
                          .format(stage, im_info['type_im'], im_info['cn'], im_info['dsmooth'], time_after - time_before))
        time.sleep(3)   # give some time to write all images properly
    else:
        logging.debug('SyntheticDeformation: all downsampled images found requested data='+im_info['data']+' stage={}, TypeIm={}, CN={}, dsmooth={}.'
                      .format(requested_stage, im_info['type_im'], im_info['cn'], im_info['dsmooth']))

    if not generation_only:
        im_stage = sitk.GetArrayFromImage(sitk.ReadImage(im_stage_address))
        deformed_im_stage = sitk.GetArrayFromImage(sitk.ReadImage(deformed_im_stage_address))
        dvf_stage = sitk.GetArrayFromImage(sitk.ReadImage(dvf_stage_address))

        if setting['torsoMask']:
            torso_stage = (sitk.GetArrayFromImage(sitk.ReadImage(torso_stage_address))).astype(np.bool)
            deformed_torso_stage = sitk.GetArrayFromImage(sitk.ReadImage(deformed_torso_stage_address))
            im_stage[torso_stage == 0] = setting['data'][im_info['data']]['defaultPixelValue']
            deformed_im_stage[deformed_torso_stage == 0] = setting['data'][im_info['data']]['defaultPixelValue']
        else:
            torso_stage = None

        im_pad = setting['ImPad_S' + str(requested_stage)]
        if im_pad > 0:
            im_stage = np.pad(im_stage, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['defaultPixelValue'],))
            deformed_im_stage = np.pad(deformed_im_stage, im_pad, 'constant', constant_values=(setting['data'][im_info['data']]['defaultPixelValue'],))

        dvf_pad = setting['DVFPad_S' + str(requested_stage)]
        if dvf_pad > 0:
            if setting['torsoMask']:
                torso_stage = np.pad(torso_stage, dvf_pad, 'constant', constant_values=(0,))

        return im_stage, deformed_im_stage, dvf_stage, torso_stage

    # def smooth(self, gonnaGenerateNextIm=0):
    #     if gonnaGenerateNextIm:
    #         MaxDeform = self._setting['MaxDeform_nextIm']
    #         Np = self._setting['Np_nextIm']
    #         sigmaB = self._setting['sigmaB_nextIm']
    #         Border = self._setting['Border_nextIm']
    #         Dim = '3D'          # The deformation of the nextIm is always 3D
    #     else:
    #         MaxDeform = self._setting['MaxDeform'][self._D]
    #         Np = self._setting['Np'][self._D]
    #         sigmaB = self._setting['sigmaB'][self._D]
    #         Border = self._setting['Border']
    #         Dim = self._setting['Dim']
    #
    #     DVFX = np.zeros(self._Im_.shape, dtype=np.float64)
    #     DVFY = np.zeros(self._Im_.shape, dtype=np.float64)
    #     DVFZ = np.zeros(self._Im_.shape, dtype=np.float64)
    #     BorderMask_=np.zeros(self._Im_.shape, dtype=np.bool)
    #     BorderMask_[Border:self._Im_.shape[0]-Border+1,Border:self._Im_.shape[1]-Border+1,Border:self._Im_.shape[2]-Border+1]=1
    #     if not gonnaGenerateNextIm:
    #         BorderMask_ = self.maskAndEdgeExtractor(BorderMask_)
    #
    #     i = 0
    #     IEdge = np.where((BorderMask_ > 0) ) # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
    #     while ((len(IEdge[0]) > 4) & (i < Np)):
    #         selectVoxel = int(np.random.randint(0, len(IEdge[0]) - 1, 1, dtype=np.int64))
    #
    #         z = IEdge[0][selectVoxel]
    #         y = IEdge[1][selectVoxel]
    #         x = IEdge[2][selectVoxel]
    #         if i < 2:  # We like to include zero deformation in our training set.
    #             Dx = 0
    #             Dy = 0
    #             Dz = 0
    #         else:
    #             Dx = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #             Dy = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #             Dz = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #
    #         DVFX[z, y, x] = Dx
    #         DVFY[z, y, x] = Dy
    #         if Dim == '3D':
    #             DVFZ[z, y, x] = Dz
    #         else:
    #             # Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction.
    #             DVFZ[z, y, x] = 0
    #         i += 1
    #
    #     _DeformedDVF_ = self.smooth_dvf(np.concatenate((np.expand_dims(DVFX, axis=3), np.expand_dims(DVFY, axis=3),
    #                                                     np.expand_dims(DVFZ, axis=3)), axis=3), dim_im=Dim, sigma_blur=np.repeat(sigmaB, 3), parallel_processing=self._setting['ParallelProcessing']).astype(np.float64)
    #     return _DeformedDVF_
    #
    #
    # def blob(self):
    #     MaxDeform = self._setting['MaxDeform'][self._D]
    #     Np = self._setting['Np'][self._D]
    #     sigmaB = self._setting['sigmaB'][self._D]
    #     Border = self._setting['Border']
    #     Dim = self._setting['Dim']
    #     DistanceDeform = self._setting['DistanceDeform']
    #     DistanceArea = self._setting['DistanceArea']
    #
    #     DVFX = np.zeros(self._Im_.shape, dtype=np.float64)
    #     DVFY = np.zeros(self._Im_.shape, dtype=np.float64)
    #     DVFZ = np.zeros(self._Im_.shape, dtype=np.float64)
    #     deformedArea_ = np.zeros(self._Im_.shape)
    #     BorderMask_ = np.zeros(self._Im_.shape, dtype=np.bool)
    #     BorderMask_[Border:self._Im_.shape[0] - Border + 1, Border:self._Im_.shape[1] - Border + 1, Border:self._Im_.shape[2] - Border + 1] = 1
    #     BorderMask_ = self.maskAndEdgeExtractor(BorderMask_)
    #
    #     i = 0
    #     IEdge = np.where(BorderMask_ > 0) # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
    #     if (len(IEdge[0]) == 0):
    #         logging.debug('SyntheticDeformation: We are out of points. Plz change the threshold value of Canny method!!!!! ') # Old method. only edges!
    #
    #     while ((len(IEdge[0]) > 4) & (i < Np)): # IEdge will change at the end of this while loop!
    #         selectVoxel = int(np.random.randint(0, len(IEdge[0]) - 1, 1, dtype=np.int64))
    #         z = IEdge[0][selectVoxel]
    #         y = IEdge[1][selectVoxel]
    #         x = IEdge[2][selectVoxel]
    #         if i < 2:  # We like to include zero deformation in our training set.
    #             Dx = 0
    #             Dy = 0
    #             Dz = 0
    #         else:
    #             Dx = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #             Dy = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #             Dz = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
    #
    #         DVFX[z, y, x] = Dx
    #         DVFY[z, y, x] = Dy
    #         DVFZ[z, y, x] = Dz
    #
    #         xminD = x - DistanceDeform
    #         xmaxD = x + DistanceDeform
    #         yminD = y - DistanceDeform
    #         ymaxD = y + DistanceDeform
    #         zminD = z - DistanceDeform
    #         zmaxD = z + DistanceDeform
    #
    #         if zmaxD > (self._Im_.shape[0] - 1): zmaxD = (self._Im_.shape[0] - 1)
    #         if ymaxD > (self._Im_.shape[1] - 1): ymaxD = (self._Im_.shape[1] - 1)
    #         if xmaxD > (self._Im_.shape[2] - 1): xmaxD = (self._Im_.shape[2] - 1)
    #         if zminD < 0: zminD = 0
    #         if yminD < 0: yminD = 0
    #         if xminD < 0: xminD = 0
    #         xminA = x - DistanceArea
    #         xmaxA = x + DistanceArea
    #         yminA = y - DistanceArea
    #         ymaxA = y + DistanceArea
    #         if (Dim == '3D'):
    #             zminA = z - DistanceArea
    #             zmaxA = z + DistanceArea
    #         else:
    #             zminA = z - 1
    #             zmaxA = z + 2  # This is exclusively for 2D !!!!
    #
    #         if zmaxA > (self._Im_.shape[0] - 1): zmaxA = (self._Im_.shape[0] - 1)
    #         if ymaxA > (self._Im_.shape[1] - 1): ymaxA = (self._Im_.shape[1] - 1)
    #         if xmaxA > (self._Im_.shape[2] - 1): xmaxA = (self._Im_.shape[2] - 1)
    #         if zminA < 0: zminA = 0
    #         if yminA < 0: yminA = 0
    #         if xminA < 0: xminA = 0
    #
    #         BorderMask_[zminD:zmaxD, yminD:ymaxD, xminD:xmaxD] = 0
    #         deformedArea_[zminA:zmaxA, yminA:ymaxA, xminA:xmaxA] = 1
    #         IEdge = np.where(BorderMask_ > 0)
    #         i += 1
    #     del BorderMask_
    #
    #     deformedArea = ip.arrayToSITK(deformedArea_, ImRef=self._Im)
    #     sitk.WriteImage(deformedArea, self.GetAddress('deformedArea'))
    #     _DeformedDVF_ = self.smooth_dvf(np.concatenate((np.expand_dims(DVFX, axis=3), np.expand_dims(DVFY, axis=3),
    #                                                     np.expand_dims(DVFZ, axis=3)), axis=3), dim_im=Dim, sigma_blur=np.repeat(sigmaB, 3), parallel_processing=self._setting['ParallelProcessing']).astype(np.float64)
    #     return _DeformedDVF_

    # def maskAndEdgeExtractor(self, BorderMask_):
    #     if self._setting['onEdge']:
    #         ImCannyAddress = self.GetAddress('DFolder') + 'canny' + str(self._setting['onEdge-lowerThreshold']) + '_' + str(self._setting['onEdge-upperThreshold']) + self._setting['ext']
    #         if os.path.isfile(ImCannyAddress):
    #             ImCanny = sitk.ReadImage(ImCannyAddress)
    #         else:
    #             ImCanny = sitk.CannyEdgeDetection(sitk.Cast(self._Im, sitk.sitkFloat32), lowerThreshold=self._setting['onEdge-lowerThreshold'], upperThreshold=self._setting['onEdge-upperThreshold'])
    #             sitk.WriteImage(sitk.Cast(ImCanny, sitk.sitkInt8), ImCannyAddress)
    #         ImCanny_ = sitk.GetArrayFromImage(ImCanny)
    #         BorderMask_ = np.logical_and(BorderMask_, ImCanny_)
    #         if self._setting['loadMask']:
    #             maskAddress = self.GetAddress('Mask')
    #             mask = sitk.ReadImage(maskAddress)
    #             # erosion with ndimage is 5 times faster than SimpleITK
    #             filter_dilate = sitk.BinaryDilateImageFilter()
    #             filter_dilate.SetKernelRadius(5)
    #             filter_dilate.SetForegroundValue(1)
    #             mask_erode = filter_dilate.Execute(mask)
    #             mask_ = sitk.GetArrayFromImage(mask_erode)
    #             BorderMask_ = np.logical_and(BorderMask_, mask_)
    #     return BorderMask_