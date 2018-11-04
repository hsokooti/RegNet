import copy
import logging
import os
from . import elastix_python as elxpy
import functions.setting_utils as su


def affine(setting, pair_info, parameter_name, stage=1, overwrite=False):

    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_moving = copy.deepcopy(pair_info[1])
    im_info_fixed['stage'] = stage
    im_info_moving['stage'] = stage

    affine_moved_address = su.address_generator(setting, 'MovedImAffine', pair_info=pair_info, **im_info_fixed)
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
    if setting['reg_UseMask']:
        fixed_mask_address = su.address_generator(setting, setting['MaskName_Affine'][0], **im_info_fixed)
        moving_mask_address = su.address_generator(setting, setting['MaskName_Affine'][0], **im_info_moving)
    elxpy.elastix(parameter_file=su.address_generator(setting, 'ParameterFolder', pair_info=pair_info,
                                                      **im_info_fixed) + parameter_name,
                  output_directory=su.address_generator(setting, 'AffineFolder', pair_info=pair_info, **im_info_fixed),
                  elastix_address='elastix',
                  fixed_image=su.address_generator(setting, 'originalIm', **im_info_fixed),
                  moving_image=su.address_generator(setting, 'originalIm', **im_info_moving),
                  fixed_mask=fixed_mask_address,
                  moving_mask=moving_mask_address,
                  initial_transform=initial_transform,
                  threads=setting['reg_NumberOfThreads'])


def affine_transformix_points(setting, pair_info, stage=1, overwrite=False):
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

    affine_output_points = su.address_generator(setting, 'reg_AffineOutputPoints', pair_info=pair_info, **im_info_fixed)
    if os.path.isfile(affine_output_points):
        if overwrite:
            logging.debug('Affine transformix overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug('Affine transformix skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug('Affine transformix starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))
    fixed_landmarks_point_elx_address = su.address_generator(setting, 'LandmarkPoint_elx', pair_info=pair_info, **im_info_fixed)
    affine_folder = su.address_generator(setting, 'AffineFolder', pair_info=pair_info, **im_info_fixed)
    elxpy.transformix(parameter_file=affine_folder + 'TransformParameters.0.txt',
                      output_directory=affine_folder,
                      points=fixed_landmarks_point_elx_address,
                      transformix_address='transformix',
                      threads=setting['reg_NumberOfThreads'])


def affine_transformix_torso(setting, pair_info, stage=1, overwrite=False):
    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_moving = copy.deepcopy(pair_info[1])
    im_info_fixed['stage'] = stage
    im_info_moving['stage'] = stage

    moved_torso_affine_address = su.address_generator(setting, 'MovedTorsoAffine', pair_info=pair_info, **im_info_fixed)
    if os.path.isfile(moved_torso_affine_address):
        if overwrite:
            logging.debug('Affine Torso overwriting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
        else:
            logging.debug('Affine Torso skipping... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
                pair_info[0]['cn'], pair_info[0]['type_im']))
            return 0
    else:
        logging.debug('Affine Torso starting... Data=' + pair_info[0]['data'] + ' CN = {} TypeIm = {}'.format(
            pair_info[0]['cn'], pair_info[0]['type_im']))

    affine_folder = su.address_generator(setting, 'AffineFolder', pair_info=pair_info, **im_info_fixed)
    output_directory = affine_folder + 'torso_moved/'
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    parameter_old_address = affine_folder + 'TransformParameters.0.txt'
    parameter_new_address = output_directory + 'TransformParameters.0.txt'
    with open(parameter_old_address, "r") as text_string:
        parameter = text_string.read()
        parameter = parameter.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0')
        parameter = parameter.replace('ResultImagePixelType "short"', 'ResultImagePixelType "char"')
    with open(parameter_new_address, "w") as text_string:
        text_string.write(parameter)

    elxpy.transformix(parameter_file=parameter_new_address,
                      output_directory=output_directory,
                      transformix_address='transformix',
                      input_image=su.address_generator(setting, 'originalTorso', **im_info_moving),
                      threads=setting['reg_NumberOfThreads'])
    old_moved_torso_affine_address = output_directory + 'result.mha'
    os.rename(old_moved_torso_affine_address, moved_torso_affine_address)