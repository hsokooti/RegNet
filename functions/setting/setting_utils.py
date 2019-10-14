import copy
import json
import logging
import numpy as np
import os
from pathlib import Path
import platform
import sys
from .artificial_generation_setting import load_deform_exp_setting
from .experiment_setting import load_global_step_from_predefined_list
from .experiment_setting import fancy_exp_name
from .experiment_setting import clean_exp_name


def initialize_setting(current_experiment, where_to_run=None):
    if where_to_run is None:
        where_to_run = 'Auto'
    setting = dict()
    setting['where_to_run'] = where_to_run
    setting['RootFolder'], setting['log_root_folder'], setting['DataFolder'] = root_address_generator(where_to_run=where_to_run)
    setting['current_experiment'] = current_experiment
    setting['stage'] = 1
    setting['UseLungMask'] = True          # The peaks of synthetic deformation can only be inside the mask
    setting['UseTorsoMask'] = True         # set background region to setting[DefaultPixelValue]
    setting['verbose_image'] = False       # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['WriteDVFStatistics'] = False
    setting['ParallelSearching'] = True
    setting['DVFPad_S1'] = 0
    setting['DVFPad_S2'] = 0
    setting['DVFPad_S4'] = 0
    setting['VoxelSize'] = [1, 1, 1]
    setting['data'] = dict()
    setting['DataList'] = ['SPREAD']
    setting['data']['SPREAD'] = load_data_setting('SPREAD')
    setting['Dim'] = 3   # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['Augmentation'] = False
    setting['WriteBSplineTransform'] = False
    setting['verbose'] = True           # Detailed printing
    setting['normalization'] = None     # The method to normalize the intensities: 'linear'
    return setting


def root_address_generator(where_to_run='Auto'):
    """
    choose the root folder, you can modify the addresses:
        'Auto'
        'Cluster'
        'Root'
    :param where_to_run:
    :return:
    """
    if where_to_run == 'Root':
        root_folder = './'
        data_folder = './Data/'
    elif where_to_run == 'Auto':
        if sys.platform == 'win32':
            root_folder = 'E:/PHD/Software/Project/DL/'
            data_folder = 'E:/PHD/Database/'
        else:
            raise ValueError('sys.platform is only defined in ["win32"]. Please defined new os in setting.setting_utils.root_address_generator()')
    elif where_to_run == 'Cluster':
        root_folder = '/exports/lkeb-hpc/hsokootioskooyi/Project/DL/'
        data_folder = '/exports/lkeb-hpc/hsokootioskooyi/Data/'
    else:
        raise ValueError('where_to_run is only defined in ["Root", "Auto", "Cluster"]. Please defined new os in setting.setting_utils.root_address_generator()')
    log_root_folder = root_folder + 'TB/'
    return root_folder, log_root_folder, data_folder


def load_setting_from_data_dict(setting, data_exp_dict_list):
    """
    :param setting:
    :param data_exp_dict_list:

        Load two predefined information:
        1. load the general setting of selected data with load_data_setting(selected_data)
        2. load the all settings of the deform_exp with load_deform_exp_setting(selected_deform_exp)

        Two list are also updated in order to have redundant information setting['DataList'], setting['DeformExpList']
    :return: setting
    """
    setting['DataExpDict'] = data_exp_dict_list
    setting['data'] = dict()
    setting['deform_exp'] = dict()
    data_list = []
    deform_exp_list = []
    for data_exp_dict in data_exp_dict_list:
        data_list.append(data_exp_dict['data'])
        setting['data'][data_exp_dict['data']] = load_data_setting(data_exp_dict['data'])
        if 'deform_exp' in data_exp_dict.keys():
            deform_exp_list.append(data_exp_dict['deform_exp'])
            setting['deform_exp'][data_exp_dict['deform_exp']] = load_deform_exp_setting(data_exp_dict['deform_exp'])
    setting['DataList'] = data_list
    setting['DeformExpList'] = deform_exp_list
    return setting


def load_data_setting(selected_data):
    """
    load the general setting of selected data like default pixel value and types of images (baseline, follow-up...)
    :param selected_data:
    :return:
    """
    data_setting = dict()
    if selected_data == 'SPREAD':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2                # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['Fixed', 'Moving']  # for eg: 'Fixed' or 'Moving' : actually Fixed indicates baseline and Moving indicates followup
        data_setting['expPrefix'] = 'ExpLung'        # for eg: ExpLung
        data_setting['DefaultPixelValue'] = -2048    # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = True
        data_setting['CNList'] = [i for i in range(1, 21)]

    elif selected_data == 'DIR-Lab_4D':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2              # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']     # for eg: 'Fixed' or 'Moving'
        data_setting['expPrefix'] = 'case'         # for eg: case
        data_setting['DefaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = False
        data_setting['CNList'] = [i for i in range(1, 11)]

    elif selected_data == 'DIR-Lab_COPD':
        data_setting['ext'] = '.mha'
        data_setting['ImageByte'] = 2                   # equals to sitk.sitkInt16 , we prefer not to import sitk in setting_utils
        data_setting['types'] = ['iBHCT', 'eBHCT']      # for eg: 'Fixed' or 'Moving'
        data_setting['expPrefix'] = 'copd'              # for eg: case
        data_setting['DefaultPixelValue'] = -2048       # The pixel value when a transformed pixel is outside of the image
        data_setting['VoxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = False
        data_setting['CNList'] = [i for i in range(1, 11)]
    else:
        logging.warning('warning: -------- selected_data not found')
    return data_setting


def address_generator(s, requested_address, data=None, deform_exp=None, type_im=0, cn=1, dsmooth=0, print_mode=False,
                      dvf_pad=None, stage=None, stage_list=None, train_mode='', c=0, semi_epoch=0, chunk=0,
                      root_folder=None, plot_mode=None, plot_itr=None, plot_i=None, current_experiment=None,
                      step=None, pair_info=None, deformed_im_ext=None, im_list_info=None, ishuffled_exp=None,
                      padto=None, bspline_folder=None, spacing=None, dvf_threshold_list=None, base_reg=None):
    if data is None:
        data = s['DataList'][0]
    if deform_exp is None:
        deform_exp = ''
        if len(s.get('DeformExpList', [])) > 0:
            deform_exp = s['DeformExpList'][0]
    if current_experiment is None:
        current_experiment = s.get('current_experiment', None)
    if '/' in current_experiment:
        log_sub_folder = current_experiment.rsplit('/')[0]
        current_experiment = current_experiment.rsplit('/')[1]
    else:
        log_sub_folder = 'RegNet'
    if root_folder is None:
        root_folder = s.get('RootFolder', None)
    deform_exp_folder = root_folder+'Elastix/Artificial_Generation/'+deform_exp+'/'+data+'/'

    if stage is None:
        stage = s.get('stage', None)
    if stage_list is None:
        if 'stage_list' in s.keys():
            stage_list = s['stage_list']

    read_pair_mode = s.get('read_pair_mode', 'real')
    if base_reg is None:
        base_reg = s.get('BaseReg', '')

    # if read_pair_mode == 'synthetic':
    #     if requested_address == 'MovedIm':
    #         requested_address = 'MovedIm_AG'

    if dvf_pad is None:
        dvf_pad = s.get('DVFPad_S' + str(stage), None)

    if bspline_folder is None:
        bspline_folder = s.get('Reg_BSpline_Folder', None)

    exp_prefix = s['data'][data]['expPrefix'] + str(cn)

    if requested_address in ['Im', 'Torso', 'Lung']:
        if dsmooth == 0:
            requested_address = 'Original' + requested_address
            if spacing == 'Raw':
                requested_address = requested_address + 'Raw'

        elif dsmooth > 0:
            requested_address = 'Next' + requested_address
    ext = s['data'][data]['ext']
    type_im_name = s['data'][data]['types'][type_im]
    address = {}
    name_dic = {}
    if data == 'SPREAD':
        if requested_address in ['OriginalFolder', 'OriginalIm', 'OriginalLung', 'OriginalTorso', 'DilatedLandmarksIm']:
            name_dic['OriginalIm'] = type_im_name + 'ImageFullRS1'
            name_dic['OriginalLung'] = type_im_name + 'MaskFullRS1'
            name_dic['OriginalTorso'] = type_im_name + 'TorsoFullRS1'
            name_dic['DilatedLandmarksIm'] = 'DilatedLandmarksFullRS1'
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address['OriginalFolder'] = root_folder + 'Elastix/LungExp/' + exp_prefix + '/Result/'
            if requested_address != 'OriginalFolder':
                address[requested_address] = address['OriginalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['OriginalImNonIsotropic', 'OriginalLandmarkFolder', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx', 'UnsurePoints']:
            patient_case = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
                            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
                            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
            type_im_landmark_tr = [patient_case[cn-1] + '_baseline_1_Cropped_point_trunc.txt',
                                   'Consensus/' + patient_case[cn-1][0:4] + '_b1f1_point_trunc.txt']
            type_im_landmark_elx = [patient_case[cn-1] + '_baseline_1_Cropped_point.txt',
                                    'Consensus/' + patient_case[cn-1][0:4] + '_b1f1_point.txt']

            address['OriginalLandmarkFolder'] = s['DataFolder'] + 'lung_dataset/SPREAD/SPREADgroundTruth/'
            original_names = ['baseline_1.mhd', 'followup_1.mhd']
            address['OriginalImNonIsotropic'] = s['DataFolder']+'lung_dataset/SPREAD/'+patient_case[cn-1]+'/'+original_names[type_im]
            address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder'] + type_im_landmark_tr[type_im]
            address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder'] + type_im_landmark_elx[type_im]
            address['UnsurePoints'] = address['OriginalLandmarkFolder'] + 'Consensus/' + patient_case[cn - 1][0:4] + '_b1f1_unsure.txt'

    elif data in ['DIR-Lab_4D', 'DIR-Lab_COPD']:
        if data == 'DIR-Lab_4D':
            dir_lab_folder = '4DCT'
        else:
            dir_lab_folder = 'COPDgene'
        if requested_address in ['OriginalImNonIsotropic', 'OriginalImNonIsotropicFolder']:
            address['OriginalImNonIsotropicFolder'] = s['DataFolder'] + 'DIR-Lab/'+dir_lab_folder+'/mha/' + exp_prefix + '/'
            address['OriginalImNonIsotropic'] = address['OriginalImNonIsotropicFolder']+exp_prefix+'_' +\
                type_im_name + ext
        elif requested_address in ['OriginalFolder', 'OriginalIm', 'OriginalImRaw', 'OriginalLung', 'OriginalLungRaw',
                                   'OriginalTorso', 'OriginalTorsoRaw', 'DilatedLandmarksIm', 'DilatedLandmarksImNonIsotropic']:

            address['OriginalFolder'] = s['DataFolder'] + 'DIR-Lab/'+dir_lab_folder+'/mha/'+exp_prefix+'/'
            name_dic['OriginalIm'] = exp_prefix + '_' + type_im_name + '_RS1'
            name_dic['OriginalImRaw'] = exp_prefix + '_' + type_im_name
            name_dic['OriginalLung'] = 'Lung_Filled/' + exp_prefix + '_' + type_im_name + '_Lung_Filled_RS1'
            name_dic['OriginalLungRaw'] = 'Lung_Filled/' + exp_prefix + '_' + type_im_name + '_Lung_Filled'
            name_dic['OriginalTorso'] = 'Torso/' + exp_prefix + '_' + type_im_name + '_Torso_RS1'
            name_dic['OriginalTorsoRaw'] = 'Torso/' + exp_prefix + '_' + type_im_name + '_Torso'
            name_dic['DilatedLandmarksIm'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name + '_Landmarks_RS1'
            name_dic['DilatedLandmarksImNonIsotropic'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            if requested_address != 'OriginalFolder':
                address[requested_address] = address['OriginalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['OriginalLandmarkFolder', 'LandmarkIndex', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx']:

            if data == 'DIR-Lab_4D':
                address['OriginalLandmarkFolder'] = s['DataFolder']+'DIR-Lab/'+dir_lab_folder+'/points/'+exp_prefix+'/'
                if ((pair_info[0]['type_im'] == 0 and pair_info[1]['type_im'] == 5) or
                    (pair_info[0]['type_im'] == 5 and pair_info[1]['type_im'] == 0)):
                        # pair_info[0]['cn'] <= 5 and pair_info[1]['cn'] <= 5:
                    address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz.txt'
                    address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_tr.txt'
                    address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_elx.txt'
                    address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_tr.txt'
                    address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_elx.txt'
                else:
                    address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz.txt'
                    address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_tr.txt'
                    address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_elx.txt'
                    address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_tr.txt'
                    address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_elx.txt'

            if data == 'DIR-Lab_COPD':
                address['OriginalLandmarkFolder'] = s['DataFolder']+'DIR-Lab/'+dir_lab_folder+'/points/'+exp_prefix+'/'
                type_im_landmark = s['data'][data]['types'][type_im][0:3]
                # address['LandmarkIndex'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_name+'_r1_xyz.txt'
                address['LandmarkIndex_tr'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_xyz_r1_tr.txt'
                address['LandmarkIndex_elx'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_xyz_r1_elx.txt'
                address['LandmarkPoint_tr'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_world_r1_tr.txt'
                address['LandmarkPoint_elx'] = address['OriginalLandmarkFolder']+'copd'+str(cn)+'_300_'+type_im_landmark+'_world_r1_elx.txt'
    if requested_address in ['ParameterFolder']:
        address['ParameterFolder'] = root_folder + 'Elastix/Registration/Parameter/'

    if requested_address in ['BaseRegFolder', 'MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg', 'Reg_BaseReg_OutputPoints', 'TransformParameterBaseReg']:

        subfolder_base_reg = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
                           pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
        address['BaseRegFolder'] = root_folder+'Elastix/Registration/'+base_reg+'/'+data+'/'+subfolder_base_reg
        address['Reg_BaseReg_OutputPoints'] = address['BaseRegFolder'] + 'outputpoints.txt'
        address['TransformParameterBaseReg'] = address['BaseRegFolder'] + 'TransformParameters.0.txt'

        if requested_address in ['MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg']:
            name_dic['MovedImBaseReg'] = 'result.0'
            name_dic['MovedTorsoBaseReg'] = 'MovedTorso'+base_reg
            name_dic['MovedLungBaseReg'] = 'MovedLung'+base_reg
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address[requested_address] = address['BaseRegFolder'] + name_dic[requested_address] + ext

    # if requested_address in ['BaseRegFolder', 'MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg', 'Reg_BaseReg_OutputPoints']:
    #     subfolder_base_reg = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
    #                        pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
    #     address['BaseRegFolder'] = root_folder+'Elastix/Registration/'+base_reg+'/'+data+'/'+subfolder_base_reg
    #     address['Reg_BaseReg_OutputPoints'] = address['BaseRegFolder'] + 'outputpoints.txt'
    #
    #     if requested_address in ['MovedImBaseReg', 'MovedTorsoBaseReg', 'MovedLungBaseReg']:
    #         name_dic['MovedImBaseReg'] = 'result.0'
    #         name_dic['MovedTorsoBaseReg'] = 'MovedTorso'+base_reg
    #         name_dic['MovedLungBaseReg'] = 'MovedLung'++base_reg
    #         if stage > 1:
    #             name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
    #         address[requested_address] = address['BaseRegFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['BSplineFolder', 'MovedImBSpline', 'BSplineOutputParameter', 'DVFBSpline', 'DVFBSpline_Jac', 'Reg_BSpline_OutputPoints']:
        subfolder_bspline = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_'+pair_info[0]['spacing']+'_' +\
                           pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'_'+pair_info[1]['spacing']+'/'
        address['BSplineFolder'] = root_folder+'Elastix/Registration/BSpline/'+data+'/'+bspline_folder+'/'+subfolder_bspline
        address['MovedImBSpline'] = address['BSplineFolder'] + 'result.0' + ext
        address['BSplineOutputParameter'] = address['BSplineFolder'] + 'TransformParameters.0.txt'
        address['DVFBSpline'] = address['BSplineFolder'] + 'deformationField' + ext
        address['DVFBSpline_Jac'] = address['BSplineFolder'] + 'Jac' + ext
        address['Reg_BSpline_OutputPoints'] = address['BSplineFolder'] + 'outputpoints.txt'

    elif requested_address in ['NextFolder', 'NextIm', 'NextLung', 'NextTorso', 'NextDVF', 'NextJac', 'NextBSplineTransform', 'NextBSplineTransformIm']:
        address['NextFolder'] = deform_exp_folder+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if print_mode:
            address['NextFolder'] = deform_exp+'/'+data+'/'+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if requested_address in ['NextBSplineTransform']:
            address[requested_address] = address['NextFolder'] + 'NextBSplineTransform.tfm'
        if requested_address in ['NextIm', 'NextLung', 'NextTorso', 'NextDVF', 'NextJac', 'NextBSplineTransformIm']:
            name_dic[requested_address] = requested_address
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address[requested_address] = address['NextFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['DSmoothFolder', 'DFolder', 'DeformedIm', 'DeformedOccluded', 'DeformedDVF', 'DeformedDVFLabel', 'DeformedArea', 'DeformedTorso',
                               'DeformedLung', 'DeformedLungOccluded', 'DVF_histogram', 'Jac', 'Jac_histogram', 'ImCanny', 'BSplineTransform',
                               'BSplineTransformIm', 'MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
        dsmooth_mod = dsmooth % len(s['deform_exp'][deform_exp]['DeformMethods'])
        deform_number = get_deform_number_from_dsmooth(s, dsmooth, deform_exp=deform_exp)
        address['DSmoothFolder'] = deform_exp_folder+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
        address['DFolder'] = address['DSmoothFolder']+s['deform_exp'][deform_exp]['DeformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'
        if print_mode:
            address['DSmoothFolder'] = deform_exp+'/'+data+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
            address['DFolder'] = address['DSmoothFolder']+s['deform_exp'][deform_exp]['DeformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'

        if requested_address in ['BSplineTransform']:
            address[requested_address] = address['DFolder'] + 'BSplineTransform.tfm'

        if requested_address in ['DeformedIm', 'DeformedOccluded', 'DeformedDVF', 'DeformedDVFLabel', 'DeformedArea', 'DeformedTorso', 'DeformedLung', 'DeformedLungOccluded',
                                 'ImCanny', 'BSplineTransformIm', 'MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
            if requested_address == 'ImCanny':
                name_dic['ImCanny'] = 'canny' + str(s['deform_exp'][deform_exp]['Canny_LowerThreshold']) + '_' + str(s['deform_exp'][deform_exp]['Canny_UpperThreshold'])
            elif requested_address in ['DeformedIm']:
                if deformed_im_ext is not None:
                    if isinstance(deformed_im_ext, list):
                        deformed_im_ext_string = deformed_im_ext[0]
                        if len(deformed_im_ext) > 1:
                            for deform_exp_i in deformed_im_ext[1:]:
                                deformed_im_ext_string = deformed_im_ext_string + '_' + deform_exp_i
                    else:
                        deformed_im_ext_string = deformed_im_ext
                    name_dic['DeformedIm'] = 'DeformedImage_' + deformed_im_ext_string
                else:
                    name_dic['DeformedIm'] = 'DeformedImage'
            elif requested_address in ['DeformedDVF', 'DeformedDVFLabel']:
                if requested_address == 'DeformedDVFLabel':
                    name_dic['DeformedDVFLabel'] = 'DeformedDVFLabel'
                    for dvf_threshold in dvf_threshold_list:
                        name_dic['DeformedDVFLabel'] = name_dic['DeformedDVFLabel']+'_'+str(dvf_threshold)
                elif requested_address == 'DeformedDVF':
                    name_dic['DeformedDVF'] = 'DeformedDVF'
                name_dic[requested_address] = name_dic[requested_address] + '_pad' + str(dvf_pad)
            else:
                name_dic[requested_address] = requested_address
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            elif requested_address in ['MovedIm_AG', 'MovedLung_AG', 'MovedTorso_AG']:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            address[requested_address] = address['DFolder'] + name_dic[requested_address] + ext

        if requested_address in ['DVF_histogram', 'Jac', 'Jac_histogram']:
            name_dic[requested_address] = requested_address + '_pad' + str(dvf_pad)
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if padto is not None:
                name_dic[requested_address] = name_dic[requested_address] + '_p' + str(padto)
            if requested_address in ['DVF_histogram', 'Jac_histogram']:
                address[requested_address] = address['DFolder'] + name_dic[requested_address] + '.png'
            elif requested_address in ['Jac']:
                address[requested_address] = address['DFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['IShuffledFolder', 'IShuffledSetting', 'IShuffled', 'IShuffledName']:
        if requested_address in ['IShuffledFolder', 'IShuffledSetting', 'IShuffled']:
            ishuffled_root_folder_name = ''
            for my_dict in s['DataExpDict']:
                ishuffled_root_folder_name = ishuffled_root_folder_name+my_dict['data']+'_'+my_dict['deform_exp']+'_'
            ishuffled_root_folder = root_folder+'Elastix/Artificial_Generation/IShuffled/IShuffled_'+ishuffled_root_folder_name[:-1]+'/'
            ishuffled_folder_name = train_mode+'_images'+str(len(im_list_info))+'_S'+str(stage)+'_exp' + str(ishuffled_exp)
            address['IShuffledFolder'] = ishuffled_root_folder + ishuffled_folder_name + '/'
            address['IShuffledSetting'] = address['IShuffledFolder'] + 'IShuffled.setting'

        if requested_address in ['IShuffled', 'IShuffledName']:
            address['IShuffledName'] = 'SemiEpoch'+str(semi_epoch)+'_Chunk'+str(chunk)+'.npy'

        if requested_address in ['IShuffled']:
            address['IShuffled'] = address['IShuffledFolder'] + address['IShuffledName']

    elif requested_address in ['IndexFolder', 'IClassFolder', 'IClass', 'IClassName']:
        address['IndexFolder'] = deform_exp_folder + 'Index' + '_S' + str(stage) + '/'
        address['IClassFolder'] = address['IndexFolder'] + 'IClass/'
        if requested_address in ['IClass', 'IClassName']:
            dsmooth_mod = dsmooth % len(s['deform_exp'][deform_exp]['DeformMethods'])
            deform_number = get_deform_number_from_dsmooth(s, dsmooth, deform_exp=deform_exp)
            class_balanced_plus_zero = np.r_[np.array([0]), s['ClassBalanced']]
            address['IClassName'] = deform_exp+'_'+type_im_name+'_cn'+str(cn)+'_Dsmooth'+str(dsmooth)+'_'+s['deform_exp'][deform_exp]['DeformMethods'][dsmooth_mod] +\
                '_'+'D'+str(deform_number)+'_M'+str(s['Margin'])+'_Z'+str(dvf_pad)+'_Torso'+str(int(s['UseTorsoMask']))+'_c'+'{:.1f}_'.format(
                class_balanced_plus_zero[c])+'{:.1f}'.format(class_balanced_plus_zero[c + 1])+'.npy'
            address['IClass'] = address['IClassFolder'] + address['IClassName']

    training_log_list = ['ModelFolder', 'summary_train', 'summary_test', 'summary_validation', 'LogFile', 'log_im_file', 'Plots_folder',
                         'saved_model', 'saved_model_with_step', 'plot_fig', 'log_folder']
    real_pair_log_list = ['result_folder', 'result_step_folder', 'result_detail_folder', 'result_landmarks_folder',
                          'full_reg_folder', 'dvf_s0', 'dvf_s_up', 'dvf_s0_jac', 'dvf_s0_jac_hist_plot', 'MovedIm', 'MovedLung', 'MovedTorso', 'landmarks_file',
                          'dvf_error']
    if requested_address in training_log_list+real_pair_log_list:
        address['log_folder'] = s['log_root_folder'] + log_sub_folder + '/' + current_experiment + '/'
        if step is None:
            step = load_global_step_from_predefined_list(current_experiment)
        if requested_address in training_log_list:
            address['ModelFolder'] = address['log_folder'] + 'train/Model/'
            address['summary_train'] = address['log_folder'] + 'train/'
            address['summary_test'] = address['log_folder'] + 'test/'
            address['summary_validation'] = address['log_folder'] + 'validation/'
            address['LogFile'] = address['ModelFolder'] + 'log.txt'
            address['log_im_file'] = address['ModelFolder'] + 'log_im.txt'
            address['Plots_folder'] = address['ModelFolder'] + 'Plots/'
            address['saved_model'] = address['ModelFolder'] + 'Saved/RegNet3DModel.ckpt'
            address['saved_model_with_step'] = address['saved_model'] + '-' + step
            address['plot_fig'] = address['ModelFolder']+'Plots/y_'+str(plot_mode)+'_itr'+str(plot_itr)+'_dir'+str(plot_i)+'.png'

        elif requested_address in real_pair_log_list:
            address['result_folder'] = address['log_folder'] + 'Results/'
            stage_step_folder = ''
            for stage_str in stage_list:
                stage_step_folder = stage_step_folder + 'S' + str(stage_str) + '_'
            stage_step_folder = stage_step_folder + 'step_' + str(step) + '/'

            if read_pair_mode == 'real':
                address['result_step_folder'] = address['result_folder']+stage_step_folder
            elif read_pair_mode == 'synthetic':
                address['result_step_folder'] = address['result_folder']+data+'/'+deform_exp+'/'+stage_step_folder

            address['result_landmarks_folder'] = address['result_step_folder'] + 'Landmarks/'
            address['landmarks_file'] = address['result_landmarks_folder'] + current_experiment + '_' + base_reg + '-' + str(step) + '.pkl'

            if requested_address == 'result_detail_folder':
                address['result_detail_folder'] = address['result_landmarks_folder'] + pair_info[0]['data'] +\
                                                  '_TypeIm'+str(pair_info[0]['type_im'])+'_TypeIm'+str(pair_info[1]['type_im'])+'/'

            if requested_address in ['full_reg_folder', 'dvf_s0', 'dvf_s0_jac', 'dvf_s0_jac_hist_plot', 'dvf_s_up', 'MovedIm', 'MovedLung', 'MovedTorso', 'dvf_error']:
                address['full_reg_folder'] = address['result_step_folder'] + 'Registration/' + base_reg + '/' +\
                                             pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_' +\
                                             pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'/'

                address['dvf_s0'] = address['full_reg_folder'] + 'DVF_S0' + ext
                address['dvf_s0_jac'] = address['full_reg_folder'] + 'DVF_S0_Jac' + ext
                address['dvf_s0_jac_hist_plot'] = address['full_reg_folder'] + 'DVF_S0_Jac_Hist.png'

                address['dvf_s_up'] = address['full_reg_folder'] + 'DVF_S' + str(stage) + '_up' + ext
                address['MovedIm'] = address['full_reg_folder'] + 'MovedImage_S' + str(stage) + ext
                address['MovedLung'] = address['full_reg_folder'] + 'MovedLung_S' + str(stage) + ext
                address['MovedTorso'] = address['full_reg_folder'] + 'MovedTorso_S' + str(stage) + ext
                if requested_address == 'dvf_error':
                    address['dvf_error'] = address['full_reg_folder'] + 'DVF_Error_' + base_reg + ext

    return address[requested_address]


def get_deform_number_from_dsmooth(setting, dsmooth, deform_exp=None):
    if deform_exp is None:
        deform_exp = setting['DeformExpList'][0]
    deform_methods = setting['deform_exp'][deform_exp]['DeformMethods']
    dsmooth_mod = dsmooth % len(deform_methods)
    selected_deform_method = deform_methods[dsmooth_mod]
    deform_method_indices = (np.where(np.array(deform_methods) == selected_deform_method))[0]
    deform_number = np.where(deform_method_indices == dsmooth_mod)[0][0]
    return int(deform_number)


def load_network_setting(setting, network_name):
    """
    load general setting by network_name.
    :param setting: 
    :param network_name: 
    
    :return: 
        setting['R']:   Radius of normal resolution patch size. Total size is (2*R +1)
        setting['Ry']:  Radius of output. Total size is (2*Ry +1)
        setting['ImPad_Sx']: Pad images with setting['DefaultPixelValue']
        setting['Margin']: Margin from the border to select random patches in the DVF numpy array, not Im numpy array
        setting['NetworkDesign]: network name, in order to keep it in the setting dict
        
    """
    import functions.network as network
    setting['NetworkDesign'] = network_name
    setting['R'], setting['Ry'] = getattr(getattr(network, network_name), 'raidus_train')()
    setting['ImPad_S'+str(setting['stage'])] = setting['R']-setting['Ry']
    setting['Margin'] = setting['Ry'] + 1
    return setting


def get_im_info_list_from_train_mode(setting, train_mode, load_mode='Single', read_pair_mode=None, stage=None):
    """
    :param setting:
    :param train_mode: should be in ['Training', 'Validation', 'Testing']
    :param load_mode: 'Single': mostly used in synthetic images, so you only need no know one image the other one will be generated.
                      'Pair'  : mostly used in real pair
                      default value is 'Single'
    :return: im_info_list:
                    load_model='Single': A list of dictionaries with single image information including:
                                        'data', 'deform_exp', 'type_im', 'cn', 'dsmooth', 'deform_method', 'deform_number'
                    load_model='Pair': Two list of dictionaries with information including:
                                        'data', 'type_im', 'cn'
    """
    if train_mode not in ['Training', 'Validation', 'Testing']:
        raise ValueError("train_mode should be in ['Training', 'Validation', 'Testing'], but it is set to"+train_mode)

    clean_data_exp_dict = []
    for data_exp in setting['DataExpDict']:
        dict_general = dict()
        for key in data_exp.keys():
            if key in ['data', 'deform_exp']:
                dict_general[key] = copy.deepcopy(data_exp[key])
            elif train_mode in key:
                key_new = key.replace(train_mode, '')
                dict_general[key_new] = copy.deepcopy(data_exp[key])
        clean_data_exp_dict.append(dict_general)

    im_info_list = []
    if load_mode == 'Single':
        for data_dict in clean_data_exp_dict:
            for cn in data_dict['CNList']:
                for type_im in data_dict['TypeImList']:
                    for dsmooth in data_dict['DSmoothList']:
                        im_info_dict = {'data': data_dict['data'], 'type_im': type_im, 'cn': cn}
                        if 'DeformMethods' in setting['deform_exp'][data_dict['deform_exp']].keys():
                            deform_methods = copy.deepcopy(setting['deform_exp'][data_dict['deform_exp']]['DeformMethods'])
                            deform_method = deform_methods[dsmooth % len(deform_methods)]
                            deform_number = get_deform_number_from_dsmooth(setting, dsmooth, deform_exp=data_dict['deform_exp'])
                            im_info_dict['deform_exp'] = data_dict['deform_exp']
                            im_info_dict['dsmooth'] = dsmooth
                            im_info_dict['deform_method'] = deform_method
                            im_info_dict['deform_number'] = deform_number

                        if 'DeformedImExt'in data_dict.keys():
                            im_info_dict['deformed_im_ext'] = data_dict['DeformedImExt']

                        if 'stage' in setting.keys() or stage is not None:
                            if stage is None:
                                stage = setting['stage']
                            im_info_dict['stage'] = stage
                            if 'PadTo' in setting.keys():
                                if 'stage'+str(stage) in setting['PadTo'].keys():
                                    im_info_dict['padto'] = setting['PadTo']['stage'+str(stage)]
                        if 'Spacing' in data_dict.keys():
                            im_info_dict['spacing'] = data_dict['Spacing']
                        im_info_list.append(im_info_dict)
    elif load_mode == 'Pair':
        if read_pair_mode is None:
            read_pair_mode = setting['read_pair_mode']

        if read_pair_mode == 'real':
            for data_dict in clean_data_exp_dict:
                for cn in data_dict['CNList']:
                    for pair in data_dict['PairList']:
                        pair_dict = [{'data': data_dict['data'], 'type_im': copy.copy(pair[0]), 'cn': cn},
                                     {'data': data_dict['data'], 'type_im': copy.copy(pair[1]), 'cn': cn}]
                        if 'Spacing' in data_dict:
                            pair_dict[0]['spacing'] = copy.copy(data_dict['Spacing'])
                            pair_dict[1]['spacing'] = copy.copy(data_dict['Spacing'])
                        im_info_list.append(pair_dict)

        elif read_pair_mode == 'synthetic':
            for data_dict in clean_data_exp_dict:
                for cn in data_dict['CNList']:
                    for type_im in data_dict['TypeImList']:
                        for dsmooth in data_dict['DSmoothList']:
                            im_info_moving = {'data': data_dict['data'], 'type_im': type_im, 'cn': cn}
                            if 'DeformMethods' in setting['deform_exp'][data_dict['deform_exp']].keys():
                                deform_methods = copy.deepcopy(setting['deform_exp'][data_dict['deform_exp']]['DeformMethods'])
                                deform_method = deform_methods[dsmooth % len(deform_methods)]
                                deform_number = get_deform_number_from_dsmooth(setting, dsmooth, deform_exp=data_dict['deform_exp'])
                                im_info_moving['deform_exp'] = data_dict['deform_exp']
                                im_info_moving['dsmooth'] = dsmooth
                                im_info_moving['deform_method'] = deform_method
                                im_info_moving['deform_number'] = deform_number

                            if 'DeformedImExt' in data_dict.keys():
                                im_info_moving['deformed_im_ext'] = data_dict['DeformedImExt']
                            if 'stage' in setting.keys() or stage is not None:
                                if stage is None:
                                    stage = setting['stage']
                                im_info_moving['stage'] = stage
                                if 'PadTo' in setting.keys():
                                    if 'stage' + str(stage) in setting['PadTo'].keys():
                                        im_info_moving['padto'] = setting['PadTo']['stage' + str(stage)]
                            if 'Spacing' in data_dict.keys():
                                im_info_moving['spacing'] = data_dict['Spacing']

                            im_info_fixed = copy.deepcopy(im_info_moving)
                            # im_info_fixed = {'data': data_dict['data'], 'type_im': type_im, 'cn': cn}
                            # if 'Spacing' in data_dict:
                            #     im_info_fixed['spacing'] = copy.copy(data_dict['Spacing'])
                            pair_dict = [im_info_fixed, im_info_moving]
                            im_info_list.append(pair_dict)

    else:
        raise ValueError("load_mode should be in ['Single', 'Pair'], but it is set to"+train_mode)
    return im_info_list


def get_pair_info_list_from_train_mode_random(setting, train_mode, stage, load_mode='Single'):
    """
    needs to be updated in future
    :param setting:
    :param train_mode: 'Training', ' Validation', 'Testing', 'Training+Validation'
    :param stage:
    :return:
    """
    if train_mode in ['Training', 'Training+Validation']:
        pair_info_training_list = get_im_info_list_from_train_mode(setting, train_mode='Training', load_mode=load_mode, stage=stage)
        random_state = np.random.RandomState(0)
        pair_info_list_copy = copy.deepcopy(pair_info_training_list)
        random_indices = random_state.permutation(len(pair_info_list_copy))
        pair_info_training_list = [pair_info_list_copy[i] for i in random_indices]
        if train_mode == 'Training':
            pair_info_list = pair_info_training_list

    if train_mode in ['Validation', 'Training+Validation']:
        pair_info_validation_list = get_im_info_list_from_train_mode(setting, train_mode='Validation', load_mode=load_mode, stage=stage)
        random_state = np.random.RandomState(0)
        pair_info_list_copy = copy.deepcopy(pair_info_validation_list)
        random_indices = random_state.permutation(len(pair_info_list_copy))
        pair_info_validation_list = [pair_info_list_copy[i] for i in random_indices[0:setting['NetworkValidation']['NumberOfImagesPerChunk']]]
        if train_mode == 'Validation':
            pair_info_list = pair_info_validation_list
        else:
            pair_info_list = pair_info_training_list + pair_info_validation_list

    if train_mode == 'Testing':
        pair_info_list = get_im_info_list_from_train_mode(setting, train_mode='Testing', load_mode=load_mode, stage=stage)

    if setting['reverse_order']:
        pair_info_list = pair_info_list[::-1]

    return pair_info_list


def load_suggested_class_balanced(setting):
    max_deform = 0
    for deform_exp in setting['DeformExpList']:
        max_deform = max(max_deform, setting['deform_exp'][deform_exp]['MaxDeform'])
    if max_deform <= 5:
        class_balanced = [max_deform]
    elif 5 < max_deform <= 7:
        class_balanced = [2, max_deform]
    elif 7 < max_deform <= 15:
        class_balanced = [1.5, 4, max_deform]
    elif 15 < max_deform <= 25:
        class_balanced = [1.5, 8, max_deform]
    else:
        raise ValueError('class_balanced are not in the defined ranges: please define the new range')
    return class_balanced


def repeat_dsmooth_numbers(dsmooth_unique_list, deform_exp, repeat):
    """
    get the dsmooth_unique_list and repeat it:
    example: assume that deform_exp['DeformMethods'] = ['respiratory_motion',
                                                        'single_frequency',
                                                        'mixed_frequency',
                                                        'zero']
    and dsmooth_unique_list = [0] which means that one type of respiratory motion is included and [1, 2, 3] is not included
    in this example andy dsmooth number with dsmooth%4 == 1 is another respiratory motion but with different seed for randomness:
    repeat_dsmooth_numbers(dsmooth_unique_list, deform_exp, 3)
            $ [0, 4, 9]

    :param dsmooth_unique_list:
    :param deform_exp:
    :param repeat:
    :return: dsmooth_list
    """
    deform_exp_dict = load_deform_exp_setting(deform_exp)
    number_of_unique_dsmooth = len(deform_exp_dict['DeformMethods'])
    dsmooth_list = []
    for r in range(repeat):
        dsmooth_list = dsmooth_list + [i+r*number_of_unique_dsmooth for i in dsmooth_unique_list]
    return dsmooth_list


def dsmoothlist_by_deform_exp(deform_exp, ag_mode):
    """
    Automatically extract the selected artificial generations for training and validation set:
        'Resp': ['respiratory_motion', 'single_frequency', 'mixed_frequency', 'zero'],
        'NoResp': ['single_frequency', 'mixed_frequency', 'zero'],
        'SingleOnly': ['single_frequency'],
        'MixedOnly': ['mixed_frequency'],
        'SingleResp': ['single_frequency', 'respiratory_motion', 'zero'],
    please note that for validation set we do not need to select all of them
    :param deform_exp:
    :param ag_mode: artificial generation mode: 'Resp', 'NoResp', 'SingleOnly', 'MixedOnly', 'SingleResp', 'Visualization'
    :return:
    """
    if ag_mode not in ['Resp', 'NoResp', 'SingleOnly', 'MixedOnly', 'SingleResp', 'Visualization']:
        raise ValueError("exp_mode should be in ['Resp', 'NoResp', 'SingleOnly', 'MixedOnly', 'SingleResp', 'Visualization']")
    dsmoothlist_training = []
    dsmoothlist_validation = []
    deform_exp_setting = load_deform_exp_setting(deform_exp)
    all_deform_methods = deform_exp_setting['DeformMethods']
    comp_dict = {'Resp': ['respiratory_motion', 'single_frequency', 'mixed_frequency', 'zero'],
                 'NoResp': ['single_frequency', 'mixed_frequency', 'zero'],
                 'SingleOnly': ['single_frequency'],
                 'MixedOnly': ['mixed_frequency'],
                 'SingleResp': ['single_frequency', 'respiratory_motion', 'zero'],
                 'Visualization': []
                 }
    for i, deform_method in enumerate(all_deform_methods):
        if deform_method in comp_dict[ag_mode]:
            dsmoothlist_training.append(i)

    if deform_exp in ['3D_max7_D14_K', '3D_max15_D14_K', '3D_max20_D14_K', '3D_max15_SingleFrequency_Visualization']:
        if ag_mode == 'Resp':
            dsmoothlist_validation = [0, 5, 10]
        elif ag_mode == 'NoResp':
            dsmoothlist_validation = [5, 8, 10]
        elif ag_mode == 'SingleResp':
            dsmoothlist_validation = [4, 8, 10]
        elif ag_mode == 'SingleOnly':
            dsmoothlist_validation = [5, 6, 8]
        elif ag_mode == 'MixedOnly':
            dsmoothlist_validation = [9, 10, 12]

    else:
        raise ValueError('dsmoothlist_validation not found for deform_exp='+deform_exp+', please add it manually')
    return dsmoothlist_training, dsmoothlist_validation


def check_setting(setting):
    for deform_exp in setting['DeformExpList']:
        if setting['ClassBalanced'][-1] > setting['deform_exp'][deform_exp]['MaxDeform']:
            raise ValueError("setting['ClassBalanced'][-1] = {} should be smaller or equal to max(setting['MaxDeform']) = {}  ".format(
                setting['ClassBalanced'][-1], max(setting['MaxDeform'])))
    if (setting['R']-setting['Ry']) < 0:
        raise ValueError("setting['R'] = {} should be greater or equal to setting['Ry'] = {}  ".format(
            setting['R'], setting['Ry']))
    if 'Randomness' in setting:
        if not setting['Randomness']:
            logging.warning('----!!!!!! setting["Randomness"] is set to False, network might not be trained correctly')
    if not setting['ParallelSearching']:
        logging.warning('----!!!!!! setting["ParallelSearching"] is set to False, running might be very slow')
    if not setting['ParallelGeneration1stEpoch']:
        logging.warning('----!!!!!! setting["ParallelGeneration1stEpoch"] is set to False, running might be very slow')

    if setting['DetailedNetworkSummary']:
        logging.warning('----!!!!!! setting["DetailedNetworkSummary"] is set to True, running might be very slow')

    im_list_info = get_im_info_list_from_train_mode(setting, train_mode='Training')
    number_of_images_last_chunk = len(im_list_info) % setting['NetworkTraining']['NumberOfImagesPerChunk']
    logging.warning('number of images in the last chunk='+str(number_of_images_last_chunk))
    if 0 < number_of_images_last_chunk < 10:
        logging.warning('----!!!! Small number of images are left for the last chunk. Total number of images in the Training='+str(len(im_list_info)) +
                        ' and NumberOfImagesPerChunk='+str(setting['NetworkTraining']['NumberOfImagesPerChunk']) +
                        ', number of images in the last chunk='+str(number_of_images_last_chunk))

    for i_dict, data_exp_dict1 in enumerate(setting['DataExpDict']):
        for key in data_exp_dict1.keys():
            if 'DeformedImExt' in key:
                for i_ext, deformed_im_ext in enumerate(data_exp_dict1[key]):
                    if deformed_im_ext not in ['Clean', 'Noise', 'Sponge', 'Occluded']:
                        raise ValueError("DeformedImExt should be in ['Clean', 'Noise', 'Sponge', 'Occluded']," +
                                         "but in data_exp_dict["+str(i_dict)+"]['"+key+"'] it is set to "+deformed_im_ext)
                    if i_ext == 0:
                        if deformed_im_ext != 'Clean':
                            raise ValueError("The first one in DeformedImExt should be 'Clean'" +
                                             "but in data_exp_dict[" + str(i_dict) + "]['" + key + "'] it is set to " + deformed_im_ext)

    if setting['NetworkDesign'] == 'crop5_connection' and setting['stage'] != 4:
        raise ValueError('in '+setting['NetworkDesign']+', the stage should be set to 4 but it is set to '+str(setting['stage']))


def write_setting(setting, setting_address=None):
    """
    Write the setting dictionary to a json file.
    :param setting:
    :param setting_address: if the setting address is not given, it choose it automatically.
           It does not overwrite if already a setting file exists in that direcotry.
           starts from 'setting0.txt' and increase the integer to find a free name.
    :return:
    """
    if setting_address is None:
        setting_folder = address_generator(setting, 'ModelFolder')
        setting_name = 'setting'
        if not os.path.isdir(setting_folder):
            os.makedirs(setting_folder)
        setting_number = 0
        setting_address = setting_folder + setting_name + str(setting_number) + '.txt'
        while os.path.isfile(setting_address):
            setting_number = setting_number + 1
            setting_address = setting_folder + setting_name + str(setting_number) + '.txt'

    with open(setting_address, 'w') as file:
        file.write(json.dumps(setting, sort_keys=True, indent=4, separators=(',', ': ')))


def load_network(setting, loaded_network):
    model_folder = address_generator(setting, 'ModelFolder', current_experiment=loaded_network['NetworkLoad'])
    setting_address = model_folder + 'setting0.txt'
    log_im_address = model_folder + 'log_im.txt'

    if 'BatchSize' in loaded_network.keys():
        if loaded_network['BatchSize'] == 'Auto':
            with open(setting_address, 'r') as f:
                setting_loaded = json.load(f)
            loaded_network['BatchSize'] = setting_loaded['NetworkTraining']['BatchSize']
            logging.info('Loading Network:'+loaded_network['NetworkLoad']+', BatchSize='+str(loaded_network['BatchSize']))

    loaded_network['GlobalStepLoad'] = get_global_step(setting, loaded_network['GlobalStepLoad'], loaded_network['NetworkLoad'])

    if 'semi_epoch_load' in loaded_network.keys():
        if loaded_network['semi_epoch_load'] == 'Auto':
            with open(log_im_address, "r") as text_string:
                log_im = text_string.read()
            semi_epoch = 0
            for line in log_im.splitlines()[::-1]:
                if 'SemiEpoch' in line:
                    semi_epoch = int((line.split('SemiEpoch=')[1]).split(',')[0])
                    break
            loaded_network['semi_epoch_load'] = semi_epoch + 1
            logging.info('Loading Network:' + loaded_network['NetworkLoad'] + ', semi_epoch_load=' + str(loaded_network['semi_epoch_load']))

    if 'itr_load' in loaded_network.keys():
        if loaded_network['itr_load'] == 'Auto':
            loaded_network['itr_load'] = int(int(loaded_network['GlobalStepLoad']) / loaded_network['BatchSize']) + 1
            logging.info('Loading Network:' + loaded_network['NetworkLoad'] + ', itr_load=' + str(loaded_network['itr_load']))

    return loaded_network


def get_global_step(setting, requested_global_step, current_experiment):
    """
    get the global step of saver in order to load the requested network model:
        'Last': search in the saved_folder to find the last network model
        'Auto': use the global step defined in the function load_global_step_from_predefined_list()
        '#Number' : otherwise the number that is requested will be used.
    :param setting:
    :param requested_global_step:
    :param current_experiment:
    :return: global_step
    """
    model_folder = address_generator(setting, 'ModelFolder', current_experiment=current_experiment)
    saved_folder = model_folder + 'Saved/'

    if requested_global_step == 'Last':
        saved_itr_list = []
        for file in os.listdir(saved_folder):
            if file.endswith('meta'):
                saved_itr_list.append(int(os.path.splitext(file.rsplit('-')[1])[0]))
        global_step = str(max(saved_itr_list))
        logging.info('Loading Network:' + current_experiment + ', GlobalStepLoad=' + global_step)

    elif requested_global_step == 'Auto':
        global_step = load_global_step_from_predefined_list(current_experiment)
        logging.info('Loading Network:' + current_experiment + ', GlobalStepLoad=' + global_step)
    else:
        global_step = requested_global_step
    return global_step
