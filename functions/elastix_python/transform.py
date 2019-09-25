import copy
import logging
import os
from . import elastix_python as elxpy
import functions.setting.setting_utils as su


def affine(setting, pair_info, parameter_name, stage=1, overwrite=False):

    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_moving = copy.deepcopy(pair_info[1])
    im_info_fixed['stage'] = stage
    im_info_moving['stage'] = stage

    affine_moved_address = su.address_generator(setting, 'MovedImBaseReg', pair_info=pair_info, **im_info_fixed)
    if os.path.isfile(affine_moved_address):
        if overwrite:
            logging.debug('Affine registration overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug('Affine registration skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug('Affine registration starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))
    fixed_mask_address = None
    moving_mask_address = None
    initial_transform = None
    if setting['Reg_BaseReg_Mask'] is not None:
        fixed_mask_address = su.address_generator(setting, setting['Reg_BaseReg_Mask'], **im_info_fixed)
        moving_mask_address = su.address_generator(setting, setting['Reg_BaseReg_Mask'], **im_info_moving)
    elxpy.elastix(parameter_file=su.address_generator(setting, 'ParameterFolder', pair_info=pair_info,
                                                      **im_info_fixed) + parameter_name,
                  output_directory=su.address_generator(setting, 'AffineFolder', pair_info=pair_info, **im_info_fixed),
                  elastix_address='elastix',
                  fixed_image=su.address_generator(setting, 'Im', **im_info_fixed),
                  moving_image=su.address_generator(setting, 'Im', **im_info_moving),
                  fixed_mask=fixed_mask_address,
                  moving_mask=moving_mask_address,
                  initial_transform=initial_transform,
                  threads=setting['Reg_NumberOfThreads'])


def bspline(setting, pair_info, parameter_name, stage=1, overwrite=False, write_dvf=True, lstm_mode=False):
    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_moving = copy.deepcopy(pair_info[1])
    im_info_fixed['stage'] = stage
    im_info_moving['stage'] = stage

    if lstm_mode:
        bspline_moved_address = su.address_generator(setting, 'MovedImBaseReg', pair_info=pair_info, base_reg=setting['Reg_BSpline_Folder'], **im_info_fixed)
    else:
        bspline_moved_address = su.address_generator(setting, 'MovedImBSpline', pair_info=pair_info, **im_info_fixed)
    if os.path.isfile(bspline_moved_address):
        if overwrite:
            logging.debug('BSpline registration overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug('BSpline registration skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug('BSpline registration starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))

    fixed_mask_address = None
    moving_mask_address = None
    if lstm_mode:
        initial_transform = su.address_generator(setting, 'TransformParameterBaseReg', pair_info=pair_info, base_reg='Affine', **im_info_fixed)
        output_directory = su.address_generator(setting, 'BaseRegFolder', pair_info=pair_info, base_reg=setting['Reg_BSpline_Folder'], **im_info_fixed)
        moving_image_address = su.address_generator(setting, 'OriginalIm', **im_info_moving)
        if setting['Reg_BSpline_Mask'] is not None:
            fixed_mask_address = su.address_generator(setting, setting['Reg_BSpline_Mask'], **im_info_fixed)
            moving_mask_address = su.address_generator(setting, setting['Reg_BSpline_Mask'], **im_info_moving)
    else:
        output_directory = su.address_generator(setting, 'BSplineFolder', pair_info=pair_info, **im_info_fixed)
        initial_transform = None
        moving_image_address = su.address_generator(setting, 'MovedImBaseReg', pair_info=pair_info, base_reg='Affine', **im_info_fixed)
        if setting['Reg_BSpline_Mask'] is not None:
            fixed_mask_address = su.address_generator(setting, setting['Reg_BSpline_Mask'], **im_info_fixed)
            moving_mask_address = su.address_generator(setting, 'Moved'+setting['Reg_BSpline_Mask']+'BaseReg',  pair_info=pair_info, base_reg='Affine', **im_info_moving)

    elxpy.elastix(parameter_file=su.address_generator(setting, 'ParameterFolder', pair_info=pair_info,
                                                      **im_info_fixed) + parameter_name,
                  output_directory=output_directory,
                  elastix_address='elastix',
                  fixed_image=su.address_generator(setting, 'OriginalIm', **im_info_fixed),
                  moving_image=moving_image_address,
                  fixed_mask=fixed_mask_address,
                  moving_mask=moving_mask_address,
                  initial_transform=initial_transform,
                  threads=setting['Reg_NumberOfThreads'])


def bsplin_transformix_dvf(setting, pair_info, stage=1, overwrite=False):
    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_fixed['stage'] = stage
    dvf_bspline_address = su.address_generator(setting, 'DVFBSpline', pair_info=pair_info, **im_info_fixed)
    if os.path.isfile(dvf_bspline_address):
        if overwrite:
            logging.debug('BSpline transformix overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug('BSpline transformix skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug('BSpline transformix starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))

    bspline_folder = su.address_generator(setting, 'BSplineFolder', pair_info=pair_info, **im_info_fixed)
    elxpy.transformix(parameter_file=su.address_generator(setting, 'BSplineOutputParameter', pair_info=pair_info, **im_info_fixed),
                      input_image=None,
                      output_directory=bspline_folder,
                      points='all',
                      threads=setting['Reg_NumberOfThreads'])


def base_reg_transformix_points(setting, pair_info, stage=1, overwrite=False, base_reg=None):
    """
    In this function we transform the points (index or world) by affine transform. This function
    utilizes transformix. However, it is also possible to read the affine parameters and do the math.
    :param setting:
    :param pair_info:
    :param stage:
    :param overwrite:
    :return:
    """
    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_fixed['stage'] = stage

    base_reg_output_points = su.address_generator(setting, 'Reg_BaseReg_OutputPoints', pair_info=pair_info, base_reg=base_reg, **im_info_fixed)
    if os.path.isfile(base_reg_output_points):
        if overwrite:
            logging.debug(base_reg+' transformix overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug(base_reg+' transformix skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug(base_reg+' transformix starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))
    fixed_landmarks_point_elx_address = su.address_generator(setting, 'LandmarkPoint_elx', pair_info=pair_info, **im_info_fixed)
    base_reg_folder = su.address_generator(setting, 'BaseRegFolder', pair_info=pair_info, base_reg=base_reg, **im_info_fixed)
    elxpy.transformix(parameter_file=base_reg_folder + 'TransformParameters.0.txt',
                      output_directory=base_reg_folder,
                      points=fixed_landmarks_point_elx_address,
                      transformix_address='transformix',
                      threads=setting['Reg_NumberOfThreads'])


def base_reg_transformix_mask(setting, pair_info, stage=1, mask_list= None, overwrite=False, base_reg=None):
    if mask_list is None:
        mask_list = ['Torso', 'Lung']

    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_moving = copy.deepcopy(pair_info[1])
    im_info_fixed['stage'] = stage
    im_info_moving['stage'] = stage

    for mask_name in mask_list:
        moved_mask_affine_address = su.address_generator(setting, 'Moved'+mask_name+'BaseReg', pair_info=pair_info, base_reg=base_reg, **im_info_fixed)
        if os.path.isfile(moved_mask_affine_address):
            if overwrite:
                logging.debug(base_reg + ' '+mask_name+' overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                    pair_info[0]['cn'], pair_info[0]['type_im']))
            else:
                logging.debug(base_reg + ' '+mask_name+' skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                    pair_info[0]['cn'], pair_info[0]['type_im']))
                continue
        else:
            logging.debug(base_reg+' '+mask_name+' starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))

        affine_folder = su.address_generator(setting, 'BaseRegFolder', pair_info=pair_info, base_reg=base_reg, **im_info_fixed)
        output_directory = affine_folder + mask_name + '_moved/'
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        parameter_old_address = affine_folder + 'TransformParameters.0.txt'
        parameter_new_address = output_directory + 'TransformParameters.0.txt'
        with open(parameter_old_address, "r") as text_string:
            parameter = text_string.read()
        parameter = parameter.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0')
        parameter = parameter.replace('ResultImagePixelType "short"', 'ResultImagePixelType "char"')

        for line in parameter.splitlines():
            if 'DefaultPixelValue' in line:
                line_default_pixel = line
                parameter = parameter.replace(line_default_pixel, '(DefaultPixelValue 0)')
                continue

        with open(parameter_new_address, "w") as text_string:
            text_string.write(parameter)

        elxpy.transformix(parameter_file=parameter_new_address,
                          output_directory=output_directory,
                          transformix_address='transformix',
                          input_image=su.address_generator(setting, 'Original'+mask_name, **im_info_moving),
                          threads=setting['Reg_NumberOfThreads'])
        old_moved_torso_affine_address = output_directory + 'result.mha'
        os.rename(old_moved_torso_affine_address, moved_mask_affine_address)


