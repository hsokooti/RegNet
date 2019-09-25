from . import utils as ag_utils
import copy
from joblib import Parallel, delayed
import logging
import multiprocessing
import numpy as np
import os
import SimpleITK as sitk
import scipy.ndimage as ndimage
import functions.image.image_processing as ip
import functions.setting.setting_utils as su


def zero(im_input_sitk):
    size_im = im_input_sitk.GetSize()[::-1]
    dvf = np.zeros(size_im+(3,))
    return dvf


def single_freq(setting, im_info, stage, im_input_sitk, gonna_generate_next_im=False):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    seed_number = ag_utils.seed_number_by_im_info(im_info, 'single_freq',
                                                  stage=stage, gonna_generate_next_im=gonna_generate_next_im)
    deform_number = im_info['deform_number']

    if gonna_generate_next_im:
        max_deform = setting['deform_exp'][im_info['deform_exp']]['NextIm_MaxDeform']
        dim_im = 3          # The deformation of the NextIm is always 3D
        seed_number = seed_number + 1
        grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_SetGridBorderToZero'][0]
        grid_spacing = setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_BSplineGridSpacing'][0]
        grid_smoothing_sigma = [i/stage for i in setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_GridSmoothingSigma'][0]]
        bspline_transform_address = su.address_generator(setting, 'NextBSplineTransform', **im_info_su)
        bspline_im_address = su.address_generator(setting, 'NextBSplineTransformIm', **im_info_su)
    else:
        max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
            setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_MaxDeformRatio'][deform_number]
        dim_im = 3
        grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_SetGridBorderToZero'][deform_number]
        grid_spacing = setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_BSplineGridSpacing'][deform_number]
        grid_smoothing_sigma = [i/stage for i in setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_GridSmoothingSigma'][deform_number]]
        bspline_transform_address = su.address_generator(setting, 'BSplineTransform', **im_info_su)
        bspline_im_address = su.address_generator(setting, 'BSplineTransformIm', **im_info_su)
    random_state = np.random.RandomState(seed_number)

    if setting['DVFPad_S'+str(stage)] > 0:
        # im_input is already zeropadded in this case
        padded_mm = setting['DVFPad_S'+str(stage)] * im_input_sitk.GetSpacing()[0]
        grid_border_to_zero = (grid_border_to_zero + np.ceil(np.repeat(padded_mm, int(dim_im)) / grid_spacing)).astype(np.int)
        if len(np.unique(im_input_sitk.GetSpacing())) > 1:
            raise ValueError('dvf_generation: padding is only implemented for isotropic voxel size. current voxel size = [{}, {}, {}]'.format(
                im_input_sitk.GetSpacing()[0], im_input_sitk.GetSpacing()[1], im_input_sitk.GetSpacing()[2]))

    bcoeff = bspline_coeff(im_input_sitk, max_deform, grid_border_to_zero, grid_smoothing_sigma,
                           grid_spacing, random_state, dim_im, artificial_generation='single_frequency')

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

    mask_to_zero = setting['deform_exp'][im_info['deform_exp']]['MaskToZero']
    if mask_to_zero is not None and not gonna_generate_next_im:
        sigma = setting['deform_exp'][im_info['deform_exp']]['SingleFrequency_BackgroundSmoothingSigma'][deform_number]
        dvf = do_mask_to_zero_gaussian(setting, im_info_su, dvf, mask_to_zero, stage, max_deform, sigma)

    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform)
    return dvf


def respiratory_motion(setting, im_info, stage, moving_image_mode='Exhale'):
    """
    Respiratory motion consists of four deformations: [2009 Hub A stochastic approach to estimate the uncertainty]
        1) Extension of the Chest in the Transversal Plane with scale of s0
        2) Decompression of the Lung in Cranio-Caudal Direction with maximum of t0
        3) Random Deformation
        4) Tissue Sliding Between Lung and Rib Cage (not implemented yet)
    :param setting:
    :param im_info:
    :param stage:
    :param moving_image_mode: 'Exhale' : mode_coeff = 1, 'Inhale': mode_coeff = -1
                               dvf[:, :, :, 2] = mode_coeff * dvf_craniocaudal
                               dvf[:, :, :, 1] = mode_coeff * dvf_anteroposterior
    :return:
    """
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    seed_number = ag_utils.seed_number_by_im_info(im_info, 'respiratory_motion', stage=stage)
    random_state = np.random.RandomState(seed_number)
    deform_number = im_info['deform_number']
    t0_max = setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_t0'][deform_number]
    s0_max = setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_s0'][deform_number]
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
        setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_MaxDeformRatio'][deform_number]
    max_deform_single_freq = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
        setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_SingleFrequency_MaxDeformRatio'][deform_number]
    grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_SetGridBorderToZero'][deform_number]
    grid_spacing = setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_BSplineGridSpacing'][deform_number]
    grid_smoothing_sigma = [i / stage for i in setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_GridSmoothingSigma'][deform_number]]

    t0 = random_state.uniform(0.8 * t0_max, 1.1 * t0_max)
    s0 = random_state.uniform(0.8 * s0_max, 1.1 * s0_max)

    if moving_image_mode == 'Inhale':
        mode_coeff = -1
    else:
        mode_coeff = 1
    im_sitk = sitk.ReadImage(su.address_generator(setting, 'Im', **im_info_su))
    lung_im = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, 'Lung', **im_info_su))).astype(np.bool)
    i_lung = np.where(lung_im)
    diaphragm_slice = np.min(i_lung[0])
    anteroposterior_dim = 1
    shift_of_center_scale = random_state.uniform(2, 12)  # in voxel
    center_scale = np.round(np.max(i_lung[anteroposterior_dim]) - shift_of_center_scale / stage)  # 10 mm above the maximum lung. will be approximately close to vertebra

    # sliding motion
    # mask_rib = im > 300
    # r = 3
    # struct = np.ones([2*r+1, 2*r+1, 2*r+1], dtype=np.bool)
    # mask_rib_close = ndimage.morphology.binary_closing(mask_rib, structure=struct)

    # slc = 50
    # import matplotlib.pyplot as plt
    # plt.figure(); plt.imshow(im[slc, :, :], cmap='gray')
    # plt.figure(); plt.imshow(mask_rib[slc, :, :])
    # plt.figure(); plt.imshow(lung_im[slc, :, :])
    # plt.figure(); plt.imshow(mask_rib_close[slc, :, :])

    logging.debug('Diaphragm slice is ' + str(diaphragm_slice))
    indices = [None] * 3
    indices[0], indices[1], indices[2] = [i * stage for i in np.meshgrid(np.arange(0, np.shape(lung_im)[0]),
                                                                         np.arange(0, np.shape(lung_im)[1]),
                                                                         np.arange(0, np.shape(lung_im)[2]),
                                                                         indexing='ij')]
    scale_transversal_plane = np.ones(np.shape(lung_im)[0])
    dvf_anteroposterior = np.zeros(np.shape(lung_im))
    dvf_craniocaudal = np.zeros(np.shape(lung_im))

    lung_extension = (np.max(i_lung[0]) - diaphragm_slice) / 2
    alpha = 1.3 / lung_extension

    for z in range(np.shape(scale_transversal_plane)[0]):
        if z < diaphragm_slice:
            scale_transversal_plane[z] = 1 + s0
            dvf_craniocaudal[z, :, :] = t0
        elif diaphragm_slice <= z < diaphragm_slice + lung_extension:
            scale_transversal_plane[z] = 1 + s0 * (1 - np.log(1 + (z - diaphragm_slice) * alpha) / np.log(1 + lung_extension * alpha))
            dvf_craniocaudal[z, :, :] = t0 * (1 - np.log(1 + (z - diaphragm_slice) * alpha) / np.log(1 + lung_extension * alpha))
        else:
            scale_transversal_plane[z] = 1
            dvf_craniocaudal[z, :, :] = 0
        dvf_anteroposterior[z, :, :] = (indices[anteroposterior_dim][z, :, :] - center_scale) * (scale_transversal_plane[z] - 1)

    dvf = np.zeros(list(np.shape(lung_im))+[3])
    dvf[:, :, :, 2] = mode_coeff * dvf_craniocaudal
    dvf[:, :, :, 1] = -mode_coeff * dvf_anteroposterior

    bcoeff = bspline_coeff(im_sitk, max_deform_single_freq, grid_border_to_zero, grid_smoothing_sigma,
                           grid_spacing, random_state, dim_im=3, artificial_generation='respiratory_motion')
    dvf_single_freq_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_single_freq_filter.SetSize(im_sitk.GetSize())
    dvf_single_freq_sitk = dvf_single_freq_filter.Execute(bcoeff)
    dvf_single_freq = sitk.GetArrayFromImage(dvf_single_freq_sitk)
    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf_single_freq = normalize_dvf(dvf_single_freq, max_deform)

    dvf_single_freq[:, :, :, 2] = dvf_single_freq[:, :, :, 2] * 0.3  # make the dvf in the slice direction smaller
    dvf = dvf + dvf_single_freq

    mask_to_zero = setting['deform_exp'][im_info['deform_exp']]['MaskToZero']
    if mask_to_zero is not None:
        sigma = setting['deform_exp'][im_info['deform_exp']]['RespiratoryMotion_BackgroundSmoothingSigma'][deform_number]
        dvf = do_mask_to_zero_gaussian(setting, im_info_su, dvf, mask_to_zero, stage, max_deform, sigma)
    else:
        raise ValueError('In the current implementation, respiratory_motion is not valid without mask_to_zero')

    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform * 1.2)

    return dvf


def mixed_freq(setting, im_info, stage):
    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage, 'padto': im_info['padto']}
    seed_number = ag_utils.seed_number_by_im_info(im_info, 'mixed_freq', stage=stage)
    random_state = np.random.RandomState(seed_number)
    deform_number = im_info['deform_number']
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
        setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_MaxDeformRatio'][deform_number]
    grid_smoothing_sigma = [i/stage for i in setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_GridSmoothingSigma'][deform_number]]
    grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_SetGridBorderToZero'][deform_number]
    grid_spacing = setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_BSplineGridSpacing'][deform_number]  # Approximately
    number_dilation = setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_Np'][deform_number]

    im_canny_address = su.address_generator(setting, 'ImCanny', **im_info_su)
    im_sitk = sitk.ReadImage(su.address_generator(setting, 'Im', **im_info_su))
    if os.path.isfile(im_canny_address):
        im_canny_sitk = sitk.ReadImage(im_canny_address)
    else:
        im_canny_sitk = sitk.CannyEdgeDetection(sitk.Cast(im_sitk, sitk.sitkFloat32),
                                                lowerThreshold=setting['deform_exp'][im_info['deform_exp']]['Canny_LowerThreshold'],
                                                upperThreshold=setting['deform_exp'][im_info['deform_exp']]['Canny_UpperThreshold'])
        sitk.WriteImage(sitk.Cast(im_canny_sitk, sitk.sitkInt8), im_canny_address)
    lung_im = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, 'Lung', **im_info_su))).astype(np.bool)
    im_canny = sitk.GetArrayFromImage(im_canny_sitk)
    # erosion with ndimage is 5 times faster than SimpleITK
    lung_dilated = ndimage.binary_dilation(lung_im)
    available_region = np.logical_and(lung_dilated, im_canny)
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
        logging.debug('dvf_generation: We are out of points. Plz change the threshold value of Canny method!!!!! ')  # Old method. only edges!
    while (len(i_edge[0][0]) > 4) and (len(i_edge[1][0]) > 4) and (len(i_edge[2][0]) > 4) and (itr_edge < number_dilation):
        # i_edge will change at the end of this while loop!
        no_more_dilatation_in_this_region = False
        for dim in range(3):
            select_voxel[dim] = int(random_state.randint(0, len(i_edge[dim][0]) - 1, 1, dtype=np.int64))
            block_low[dim], block_high[dim] = center_to_block(setting,
                                                              center=np.array([i_edge[dim][0][select_voxel[dim]],
                                                                               i_edge[dim][1][select_voxel[dim]],
                                                                               i_edge[dim][2][select_voxel[dim]]]),
                                                              radius=round(setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_BlockRadius']/stage),
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
            if random_state.uniform() > 0.3:
                no_more_dilatation_in_this_region = True
        if no_more_dilatation_in_this_region:
            available_region[block_low[dim][0]:block_high[dim][0], block_low[dim][1]:block_high[dim][1], block_low[dim][2]:block_high[dim][2], dim] = False
        if itr_edge >= np.round(10*number_dilation/12):
            for dim in range(3):
                i_edge[dim] = np.where(available_region[:, :, :, dim] > 0)
        itr_edge += 1

    bcoeff = bspline_coeff(im_sitk, max_deform, grid_border_to_zero, grid_smoothing_sigma,
                           grid_spacing, random_state, dim_im=3, artificial_generation='mixed_frequency')
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_sitk.GetSize())
    smoothed_values_sitk = dvf_filter.Execute(bcoeff)
    smoothed_values = sitk.GetArrayFromImage(smoothed_values_sitk)

    dvf = (dilated_edge.astype(np.float64) * smoothed_values).astype(np.float64)
    if setting['DVFPad_S'+str(stage)] > 0:
        pad = setting['DVFPad_S'+str(stage)]
        dvf = np.pad(dvf, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0,))

    sigma_range = setting['deform_exp'][im_info['deform_exp']]['MixedFrequency_SigmaRange'][deform_number]
    sigma = random_state.uniform(low=sigma_range[0] / stage,
                                 high=sigma_range[1] / stage,
                                 size=3)
    dvf = smooth_dvf(dvf, sigma_blur=sigma, parallel_processing=setting['ParallelSearching'])

    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform)

    return dvf


def translation(setting, im_info, stage, im_input_sitk):
    seed_number = ag_utils.seed_number_by_im_info(im_info, 'translation', stage=stage)
    random_state = np.random.RandomState(seed_number)
    deform_number = im_info['deform_number']
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
        setting['deform_exp'][im_info['deform_exp']]['Translation_MaxDeformRatio'][deform_number]
    dim_im = setting['Dim']
    translation_transform = sitk.TranslationTransform(dim_im)

    translation_magnitude = np.zeros(3)
    for dim in range(dim_im):
        if random_state.random_sample() > 0.8:
            translation_magnitude[dim] = 0
        else:
            translation_magnitude[dim] = random_state.uniform(-max_deform, max_deform)

    translation_transform.SetParameters(translation_magnitude)
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_input_sitk.GetSize())
    dvf_sitk = dvf_filter.Execute(translation_transform)
    dvf = sitk.GetArrayFromImage(dvf_sitk)

    return dvf


def bspline_coeff(im_input_sitk, max_deform, grid_border_to_zero, grid_smoothing_sigma, grid_spacing, random_state, dim_im, artificial_generation=None):
    number_of_grids = list(np.round(np.array(im_input_sitk.GetSize()) * np.array(im_input_sitk.GetSpacing()) / grid_spacing))
    number_of_grids = [int(i) for i in number_of_grids]  # This is a bit funny, it has to be int (and not even np.int)
    # BCoeff = sitk.BSplineTransformInitializer(ImInput, numberOfGrids, order=3)
    # problem with the offset
    bcoeff = sitk.BSplineTransformInitializer(sitk.Image(im_input_sitk.GetSize(), sitk.sitkInt8), number_of_grids, order=3)
    bcoeff_parameters = random_state.uniform(-max_deform*4, max_deform*4, len(bcoeff.GetParameters()))
    # we choose numbers to be in range of MaxDeform, please note that there are two smoothing steps after this initialization.
    # So numbers will be much smaller.

    grid_side = bcoeff.GetTransformDomainMeshSize()
    if dim_im == 3:
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
    else:
        raise ValueError('not implemented for 2D')
    if artificial_generation in ['single_frequency', 'respiratory_motion']:
        bcoeff_parameters_smooth_normalize = normalize_dvf(bcoeff_parameters_smooth, max_deform * 1.7)
    elif artificial_generation == 'mixed_frequency':
        bcoeff_parameters_smooth_normalize = normalize_dvf(bcoeff_parameters_smooth, max_deform * 2, min_deform=max_deform)
    else:
        raise ValueError("artificial_generation should be in ['single_frequency', 'mixed_frequency', 'respiratory_motion']")

    bcoeff.SetParameters(bcoeff_parameters_smooth_normalize)
    return bcoeff


def smooth_dvf(dvf, dim_im=3, sigma_blur=None, parallel_processing=True):
    dvf_smooth = np.empty(np.shape(dvf))
    if parallel_processing:
        num_cores = multiprocessing.cpu_count() - 2
        if dim_im == 3:
            # The following line is not working in Windows
            [dvf_smooth[:, :, :, 0], dvf_smooth[:, :, :, 1], dvf_smooth[:, :, :, 2]] = \
                Parallel(n_jobs=num_cores)(delayed(smooth_gaussian)(dvf=dvf[:, :, :, i], sigma=sigma_blur[i]) for i in range(np.shape(dvf)[3]))
        if dim_im == 2:
            [dvf_smooth[:, :, :, 0], dvf_smooth[:, :, :, 1]] = \
                Parallel(n_jobs=num_cores)(delayed(smooth_gaussian)(dvf=dvf[:, :, :, i], sigma=sigma_blur[i]) for i in range(np.shape(dvf)[3]))
            dvf_smooth[:, :, :, 2] = dvf[:, :, :, 2]
    else:
        for dim in range(dim_im):
            dvf_smooth[:, :, :, dim] = smooth_gaussian(dvf[:, :, :, dim], sigma_blur[dim])
    return dvf_smooth


def normalize_dvf(dvf, max_deform, min_deform=None):
    max_dvf = max(abs(np.max(dvf)), abs(np.min(dvf)))
    if max_dvf > max_deform:
        dvf = dvf * max_deform / max_dvf

    if min_deform is not None:
        if max_dvf < min_deform:
            dvf = dvf * min_deform / max_dvf

    return dvf


def smooth_gaussian(dvf, sigma):
    return ndimage.filters.gaussian_filter(dvf, sigma=sigma)


def center_to_block(setting, center=None, radius=10, im_ref=None):
    block_low = center - radius
    block_high = center + radius
    if setting['Dim'] == 2:
        block_low[0] = center[0] - 1
        block_high[0] = center[0] + 2
    for dim in range(3):
        if block_low[dim] < 0:
            block_low[dim] = 0
        if block_high[dim] > im_ref.GetSize()[-1-dim]:
            block_high[dim] = im_ref.GetSize()[-1-dim]
    return block_low, block_high


def do_mask_to_zero_gaussian(setting, im_info_su, dvf, mask_to_zero, stage, max_deform, sigma):
    mask_address = su.address_generator(setting, mask_to_zero, **im_info_su)
    mask_im = sitk.GetArrayFromImage(sitk.ReadImage(mask_address))
    dvf = dvf * np.repeat(np.expand_dims(mask_im, axis=3), np.shape(dvf)[3], axis=3)
    sigma = sigma / stage * max_deform / 7  # in stage 4 we should make this sigma smaller but at the same time
    sigma = np.tile(sigma, 3)
    # the max_deform in stage 4 is 20 which leads to negative jacobian. There is no problem for other sigma values in the code.
    dvf = smooth_dvf(dvf, sigma_blur=sigma, parallel_processing=setting['ParallelSearching'])
    return dvf


def background_to_zero_linear(setting, im_info_su, gonna_generate_next_im=False):
    if gonna_generate_next_im:
        im_info_su_orig = copy.deepcopy(im_info_su)
        im_info_su_orig['dsmooth'] = 0
        torso_address = su.address_generator(setting, 'Torso', **im_info_su_orig)
    else:
        torso_address = su.address_generator(setting, 'Torso', **im_info_su)

    torso_im = sitk.GetArrayFromImage(sitk.ReadImage(torso_address))
    torso_distance = ndimage.morphology.distance_transform_edt(1 - torso_im, sampling=setting['VoxelSize'])
    mask_to_zero = torso_im.copy().astype(np.float)
    background_ind = [torso_im == 0]
    mask_to_zero[background_ind] = (1 / torso_distance[background_ind])
    mask_to_zero[mask_to_zero < 0.05] = 0
    return mask_to_zero


def translation_with_bspline_grid(setting, im_input_sitk, im_info=None):
    seed_number = ag_utils.seed_number_by_im_info(im_info, 'translation')
    random_state = np.random.RandomState(seed_number)
    deform_number = im_info['deform_number']
    max_deform = setting['deform_exp'][im_info['deform_exp']]['MaxDeform'] * \
        setting['deform_exp'][im_info['deform_exp']]['Translation_MaxDeformRatio'][deform_number]
    dim_im = setting['Dim']
    grid_border_to_zero = setting['deform_exp'][im_info['deform_exp']]['setGridBorderToZero_translation'][deform_number]
    grid_spacing = setting['deform_exp'][im_info['deform_exp']]['BsplineGridSpacing_translation'][deform_number]

    if setting['DVFPad_S1'] > 0:
        # ImInput is already zeropadded in this case
        padded_mm = setting['DVFPad_S1'] * im_input_sitk.GetSpacing()[0]
        grid_border_to_zero = (grid_border_to_zero + np.ceil(np.repeat(padded_mm, int(dim_im[0])) / grid_spacing)).astype(np.int)
        if len(np.unique(im_input_sitk.GetSpacing())) > 1:
            raise ValueError('dvf_generation: padding is only implemented for isotropic voxel size. current voxel size = [{}, {}, {}]'.format(
                im_input_sitk.GetSpacing()[0], im_input_sitk.GetSpacing()[1], im_input_sitk.GetSpacing()[2]))

    number_of_grids = list(np.round(np.array(im_input_sitk.GetSize()) * np.array(im_input_sitk.GetSpacing()) / grid_spacing))
    number_of_grids = [int(i) for i in number_of_grids]  # it has to be int (and not even np.int)
    # BCoeff = sitk.BSplineTransformInitializer(ImInput, numberOfGrids, order=3)
    # problem with the offset
    bcoeff = sitk.BSplineTransformInitializer(sitk.Image(im_input_sitk.GetSize(), sitk.sitkInt8), number_of_grids, order=3)
    grid_side = bcoeff.GetTransformDomainMeshSize()
    if dim_im == 3:
        bcoeff_smoothed_dim = [None] * 3
        translation_magnitude = [None] * 3
        for dim in range(3):
            if random_state.random_sample() > 0.8:
                translation_magnitude[dim] = 0
            else:
                translation_magnitude[dim] = random_state.uniform(-max_deform, max_deform)
            if dim == 2:
                if translation_magnitude[2] < max_deform * 2 / 3:
                    if translation_magnitude[1] < max_deform * 2 / 3:
                        if translation_magnitude[0] < max_deform * 2 / 3:
                            translation_magnitude[2] = random_state.uniform(max_deform * 2 / 3, max_deform)
                            sign_of_magnitude = random_state.random_sample()
                            if sign_of_magnitude > 0.5:
                                translation_magnitude[2] = -  translation_magnitude[2]

            bcoeff_dim = np.ones([grid_side[2] + 3, grid_side[1] + 3, grid_side[0] + 3]) * translation_magnitude[dim]
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
    else:
        raise ValueError('not implemented for 2D')
    bcoeff.SetParameters(bcoeff_parameters_smooth)
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetSize(im_input_sitk.GetSize())
    dvf_sitk = dvf_filter.Execute(bcoeff)
    dvf = sitk.GetArrayFromImage(dvf_sitk)
    if setting['deform_exp'][im_info['deform_exp']]['DVFNormalization']:
        dvf = normalize_dvf(dvf, max_deform)
    return dvf
