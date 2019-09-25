import copy
import json
import logging
import numpy as np
import os
import platform
import sys


def initialize_setting(current_experiment, where_to_run=None):
    if where_to_run is None:
        where_to_run = 'Auto'
    setting = dict()
    setting['where_to_run'] = where_to_run
    setting['RootFolder'], setting['log_root_folder'], setting['dataFolder'] = root_address_generator(where_to_run=where_to_run)
    setting['current_experiment'] = current_experiment
    setting['stage'] = 0
    setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
    setting['torsoMask'] = True  # set background region to setting[defaultPixelValue]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['WriteDVFStatistics'] = False
    setting['ParallelProcessing'] = True
    setting['DVFPad_S1'] = 0
    setting['DVFPad_S2'] = 0
    setting['DVFPad_S4'] = 0
    setting['voxelSize'] = [1, 1, 1]
    setting['data'] = dict()
    setting['DataList'] = ['SPREAD']
    setting['data']['SPREAD'] = load_data_setting('SPREAD')
    setting['Dim'] = '3D'   # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['Augmentation'] = False
    setting['WriteBSplineTransform'] = False

    return setting


def root_address_generator(where_to_run='Auto'):
    if where_to_run is 'Auto':
        if sys.platform == 'win32':
            root_folder = 'E:/PHD/Software/Project/DL/'
            data_folder = 'E:/PHD/Database/'
        elif sys.platform == 'linux':
            root_folder = '/srv/' + platform.node() + '/hsokooti/DL/'
            data_folder = '/srv/' + platform.node() + '/hsokooti/Data/'
        else:
            raise ValueError('sys.platform is only defined in ["win32", "linux"]. Please defined new os in setting_utils.root_address_generator()')
    elif where_to_run == 'Cluster':
        root_folder = '/exports/lkeb-hpc/hsokootioskooyi/Project/DL/'
        data_folder = '/exports/lkeb-hpc/hsokootioskooyi/Data/'
    else:
        raise ValueError('where_to_run is only defined in ["Auto", "Cluster"]. Please defined new os in setting_utils.root_address_generator()')
    log_root_folder = root_folder + 'RegNet2/TB/3D/'
    return root_folder, log_root_folder, data_folder


def load_setting_from_data_dict(setting, data_exp_dict):
    setting['DataExpDict'] = data_exp_dict
    setting['data'] = dict()
    setting['deform_exp'] = dict()
    data_list = []
    deform_exp_list = []
    for single_dict in data_exp_dict:
        data_list.append(single_dict['data'])
        setting['data'][single_dict['data']] = load_data_setting(single_dict['data'])
        if 'deform_exp' in single_dict.keys():
            deform_exp_list.append(single_dict['deform_exp'])
            setting['deform_exp'][single_dict['deform_exp']] = load_deform_exp_setting(single_dict['deform_exp'])
    setting['DataList'] = data_list
    setting['DeformExpList'] = deform_exp_list
    return setting


def load_data_setting(selected_data):
    data_setting = dict()
    if selected_data == 'SPREAD':
        data_setting['ext'] = '.mha'
        data_setting['imageByte'] = 2                # equals to sitk.sitkInt16 , we prefer not to import sitk in SettingUtils
        data_setting['types'] = ['Fixed', 'Moving']  # for eg: 'Fixed' or 'Moving' : actually Fixed indicates baseline and Moving indicates followup
        data_setting['expPrefix'] = 'ExpLung'        # for eg: ExpLung1
        data_setting['defaultPixelValue'] = -2048    # The pixel value when a transformed pixel is outside of the image
        data_setting['voxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = True
        data_setting['CNList'] = [i for i in range(1, 21)]

    elif selected_data == 'DIR-Lab_4D':
        data_setting['Dim'] = '3D'
        data_setting['ext'] = '.mha'
        data_setting['imageByte'] = 2              # equals to sitk.sitkInt16 , we prefer not to import sitk in SettingUtils
        data_setting['types'] = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']     # for eg: 'Fixed' or 'Moving' : actually Fixed indicates baseline and Moving indicates followup
        data_setting['expPrefix'] = 'case'         # for eg: case1
        data_setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
        data_setting['voxelSize'] = [1, 1, 1]
        data_setting['AffineRegistration'] = True
        data_setting['UnsureLandmarkAvailable'] = False
        data_setting['CNList'] = [i for i in range(1, 11)]
    else:
        print('warning: -------- selected_data not found')
    return data_setting


def load_deform_exp_setting(selected_deform_exp):
    deform_exp_setting = dict()
    if selected_deform_exp == '3D_max15_D9':
        deform_exp_setting['MaxDeform'] = [15, 15, 15]  # The maximum amplitude of deformations
        deform_exp_setting['deformMethods'] = ['translation', 'translation', 'translation',
                                               'smoothBSpline', 'smoothBSpline', 'smoothBSpline',
                                               'dilatedEdgeSmooth', 'dilatedEdgeSmooth', 'dilatedEdgeSmooth']
        deform_exp_setting['DsmoothList'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
        deform_exp_setting['stage'] = 0
        deform_exp_setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
        deform_exp_setting['torsoMask'] = True  # set background region to setting[defaultPixelValue]
        deform_exp_setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
        deform_exp_setting['WriteDVFStatistics'] = True
        deform_exp_setting['ParallelProcessing'] = True
        deform_exp_setting['DVFNormalization'] = True
        deform_exp_setting['DVFPad_S1'] = 0
        deform_exp_setting['DVFPad_S2'] = 0
        deform_exp_setting['DVFPad_S4'] = 0

        deform_exp_setting['types'] = ['Fixed', 'Moving']               # for eg: 'Fixed' or 'Moving' : actually Fixed indicates baseline and Moving indicates followup
        deform_exp_setting['IndexFolderName'] = 'Index'

        # images
        deform_exp_setting['onEdge-lowerThreshold'] = 50.0
        deform_exp_setting['onEdge-upperThreshold'] = 100.0
        deform_exp_setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
        deform_exp_setting['sigmaN'] = 10  # Sigma for adding noise after deformation

        # blob
        deform_exp_setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
        deform_exp_setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
        deform_exp_setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
        deform_exp_setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
        deform_exp_setting['Np'] = [100, 100, 100]  # Number of random peaks
        deform_exp_setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)

        deform_exp_setting['Border_nextIm'] = 33
        deform_exp_setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
        deform_exp_setting['MaxDeform_nextIm'] = 15
        deform_exp_setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
        deform_exp_setting['Np_nextIm'] = 100

        # smoothBspline
        deform_exp_setting['BsplineGridSpacing_smooth'] = [[40, 40, 40], [30, 30, 30], [20, 20, 20]]  # in mm approximately
        deform_exp_setting['setGridBorderToZero'] = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]
        deform_exp_setting['GridSmoothingSigma'] = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]

        # dilatedEdge
        deform_exp_setting['blockRadius_dilatedEdge'] = 20
        deform_exp_setting['MaxDeform_dilateEdge'] = [50, 50, 50]
        deform_exp_setting['Np_dilateEdge'] = [200, 150, 150]
        deform_exp_setting['BsplineGridSpacing_dilatedEdge'] = [[80, 80, 80], [40, 40, 40], [20, 20, 80]]
        deform_exp_setting['sigmaRange_dilatedEdge'] = [[5, 10], [5, 10], [5, 10]]
        deform_exp_setting['GridSmoothingSigma_dilatedEdge'] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        deform_exp_setting['setGridBorderToZero_dilatedEdge'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # translation
        deform_exp_setting['BsplineGridSpacing_translation'] = [[40, 40, 40], [40, 40, 40], [40, 40, 40]]
        deform_exp_setting['setGridBorderToZero_translation'] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    elif selected_deform_exp == '3D_max25_D12':
        deform_exp_setting = dict()
        deform_exp_setting['Dim'] = '3D'  # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
        deform_exp_setting['MaxDeform'] = [25, 25, 25, 25]  # The maximum amplitude of deformations
        deform_exp_setting['deformMethods'] = ['translation', 'translation', 'translation', 'translation',
                                               'smoothBSpline', 'smoothBSpline', 'smoothBSpline', 'smoothBSpline',
                                               'dilatedEdgeSmooth', 'dilatedEdgeSmooth', 'dilatedEdgeSmooth', 'dilatedEdgeSmooth']
        deform_exp_setting['DsmoothList'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
        deform_exp_setting['stage'] = 0
        deform_exp_setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
        deform_exp_setting['torsoMask'] = True  # set background region to setting[defaultPixelValue]
        deform_exp_setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
        deform_exp_setting['WriteDVFStatistics'] = False
        deform_exp_setting['ParallelProcessing'] = True
        deform_exp_setting['DVFNormalization'] = True

        deform_exp_setting['DVFPad_S1'] = 0
        deform_exp_setting['DVFPad_S2'] = 0
        deform_exp_setting['DVFPad_S4'] = 0

        # images
        deform_exp_setting['onEdge-lowerThreshold'] = 50.0
        deform_exp_setting['onEdge-upperThreshold'] = 100.0
        deform_exp_setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
        deform_exp_setting['sigmaN'] = 10  # Sigma for adding noise after deformation

        # blob
        deform_exp_setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
        deform_exp_setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
        deform_exp_setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
        deform_exp_setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
        deform_exp_setting['Np'] = [100, 100, 100]  # Number of random peaks
        deform_exp_setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)

        deform_exp_setting['Border_nextIm'] = 33
        deform_exp_setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
        deform_exp_setting['MaxDeform_nextIm'] = 15
        deform_exp_setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
        deform_exp_setting['Np_nextIm'] = 100

        # smoothBspline
        deform_exp_setting['BsplineGridSpacing_smooth'] = [[40, 40, 40], [30, 30, 30], [20, 20, 20], [20, 20, 20]]  # in mm approximately
        deform_exp_setting['setGridBorderToZero'] = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        deform_exp_setting['GridSmoothingSigma'] = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]

        # dilatedEdge
        deform_exp_setting['blockRadius_dilatedEdge'] = 20
        deform_exp_setting['MaxDeform_dilateEdge'] = [50, 50, 50, 50]
        deform_exp_setting['Np_dilateEdge'] = [200, 150, 150, 150]
        deform_exp_setting['BsplineGridSpacing_dilatedEdge'] = [[80, 80, 80], [40, 40, 40], [20, 20, 80], [20, 20, 80]]
        deform_exp_setting['sigmaRange_dilatedEdge'] = [[5, 10], [5, 10], [5, 10], [5, 10]]
        deform_exp_setting['GridSmoothingSigma_dilatedEdge'] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        deform_exp_setting['setGridBorderToZero_dilatedEdge'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # translation
        deform_exp_setting['BsplineGridSpacing_translation'] = [[40, 40, 40], [40, 40, 40], [40, 40, 40], [40, 40, 40]]
        deform_exp_setting['setGridBorderToZero_translation'] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

    elif selected_deform_exp == '3D_max20_D12':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max25_D12')
        deform_exp_setting = copy.deepcopy(deform_exp_setting_temp)
        deform_exp_setting['MaxDeform'] = [20, 20, 20, 20]

    elif selected_deform_exp == '3D_max15_D12':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max25_D12')
        deform_exp_setting = copy.deepcopy(deform_exp_setting_temp)
        deform_exp_setting['MaxDeform'] = [15, 15, 15, 15]

    elif selected_deform_exp == '3D_max7_D12':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max25_D12')
        deform_exp_setting = copy.deepcopy(deform_exp_setting_temp)
        deform_exp_setting['MaxDeform'] = [7, 7, 7, 7]

    elif selected_deform_exp == '3D_max7_D9':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max15_D9')
        deform_exp_setting = copy.deepcopy(deform_exp_setting_temp)
        deform_exp_setting['MaxDeform'] = [7, 7, 7]

    elif selected_deform_exp == '3D_max15_D12_Visualization':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max15_D12')
        deform_exp_setting = copy.deepcopy(deform_exp_setting_temp)

    else:
        print('warning: -------- selected_deform_exp not found')
    return deform_exp_setting


def load_global_step_from_current_experiment(exp):
    global_step_dict = {'20180124_max20_2M': '1155015',
                        '20180204_max8': '1080020',
                        '20180205_max20': '0',
                        '20180210_max15': '940020',
                        '20180210_smallNet': '1240020',
                        '20180210_19Pair': '2140020',
                        '20180217_Np10_Small1': '1060020',
                        '20180218_Np10_Medium1': '1020020',
                        '20180223_max10_Small2': '1260020',
                        '20180224_max10_Small2_R1': '1300020',
                        '20180224_max10_Small2_R1_R2': '1620020',
                        '20180226_max15_max8_Samll2': '1620020',
                        '20180308_3Dmax15_E_max8_Small2': '1740020',
                        '20180305_max15_max8_crop1': '1620020',
                        '20180312_max15_E_max8_crop1_connection': '2370015',
                        '20180312_max15_max8_crop1_connection': '1980015',
                        '20180321_max10_crop1_connection': '2925015',
                        '20180321_max10_crop1_connection-MultiRes': '2925015-MultiRes',
                        '20180510_max10_ED_crop1_connection': '1920015',
                        '20180510_max10_ED_crop1_connection-MultiRes-Old': '1920015-MultiRes-Old',
                        '20180510_max10_ED_crop1_connection-MultiRes': '1920015-MultiRes',
                        '20180622_max7_D9_stage0': '3075015',
                        '20180622_max15_D9_stage2': '10410015',
                        '20180613_max15_D9_stage4': '12390015',
                        '20181008_max7_D9_crop3_A': '5460015',
                        '20180921_max15_D12_stage2_crop3': '10020015',
                        '20181009_max20_D12_stage4_crop3': '12630015',
                        '20181030_225202_sh': '5505015',
                        '20181030_225310_sh': '6735015',
                        '20181017_max20_S4_dec3': '8910015'
                        }

    if exp not in global_step_dict.keys():
        global_step_dict[exp] = '0'
    global_step = global_step_dict[exp]
    return global_step


def fancy_exp_name(exp):
    fancy_dict = {'Affine': 'Affine',
                  'Bspline_1S': 'Bspline 1S',
                  'Bspline_3S': 'Bspline 3S',
                  'multi_stage1b_torso_S4_S2_S1': 'RegNet-max-pooling 3S',
                  'two_stage_torso_S2_S1': 'RegNet-max 2S',
                  'one_stage_torso_S1': 'RegNet-max-pooling 1S',
                  }
    if exp not in fancy_dict.keys():
        fancy_dict[exp] = exp
    fancy_name = fancy_dict[exp]
    return fancy_name


def address_generator(s, requested_address, data=None, deform_exp=None, type_im=0, cn=1, dsmooth=0, print_mode=False,
                      dvf_pad=None, stage=None, stage_list=None, train_mode='', c=0, semi_epoch=0, number_of_images_per_chunk=0,
                      samples_per_image=0, chunk=0, root_folder=None, plot_mode=None, plot_itr=None, plot_i=None,
                      current_experiment=None, step=None, pair_info=None):
    if data is None:
        data = s['DataList'][0]
    if deform_exp is None:
        if 'DeformExpList' in s.keys():
            if 'DeformExpList' in s.keys():
                if len(s['DeformExpList']) > 0:
                    deform_exp = s['DeformExpList'][0]
                else:
                    deform_exp = ''
    if current_experiment is None:
        current_experiment = s['current_experiment']
    if root_folder is None:
        root_folder = s['RootFolder']
    if 'read_pair_mode' not in s.keys():
        read_pair_mode = 'real'
    else:
        read_pair_mode = s['read_pair_mode']

    if stage is None:
        stage = s['stage']
    if dvf_pad is None:
        if 'DVFPad_S' + str(stage) in s.keys():
            dvf_pad = s['DVFPad_S' + str(stage)]

    exp_prefix = s['data'][data]['expPrefix'] + str(cn)

    if requested_address in ['Im', 'Torso', 'Mask']:
        if dsmooth == 0:
            requested_address = 'original' + requested_address
        elif dsmooth > 0:
            requested_address = 'next' + requested_address
    ext = s['data'][data]['ext']
    type_im_name = s['data'][data]['types'][type_im]
    address = {}
    name_dic = {}
    if data == 'SPREAD':
        if requested_address in ['originalFolder', 'originalIm', 'originalMask', 'originalTorso', 'DilatedLandmarksIm']:
            name_dic['originalIm'] = type_im_name + 'ImageFullRS1'
            name_dic['originalMask'] = type_im_name + 'MaskFullRS1'
            name_dic['originalTorso'] = type_im_name + 'TorsoFullRS1'
            name_dic['DilatedLandmarksIm'] = 'DilatedLandmarksFullRS1'
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address['originalFolder'] = root_folder + 'Elastix/LungExp/' + exp_prefix + '/Result/'
            if requested_address != 'originalFolder':
                address[requested_address] = address['originalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['originalLandmarkFolder', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx', 'UnsurePoints']:
            patient_case = ['p000', 'p001', 'p003', 'p005', 'p006', 'p007', 'p008',
                            'p009', 'p011', 'p012', 'p013', 'p014', 'p015', 'p017',
                            'p018', 'p019', 'p020', 'p021', 'p022', 'p023', 'p024']
            type_im_landmark_tr = [patient_case[cn-1] + '_baseline_1_Cropped_point_trunc.txt',
                                   'Consensus/' + patient_case[cn-1][0:4] + '_b1f1_point_trunc.txt']
            address['originalLandmarkFolder'] = s['dataFolder'] + 'lung_dataset/SPREADgroundTruth/'
            address['LandmarkPoint_tr'] = address['originalLandmarkFolder'] + type_im_landmark_tr[type_im]
            address['UnsurePoints'] = address['originalLandmarkFolder'] + 'Consensus/' + patient_case[cn - 1][0:4] + '_b1f1_unsure.txt'

        elif requested_address in ['AffineFolder', 'MovedImAffine', 'MovedTorsoAffine', 'reg_AffineOutputPoints']:
            address['AffineFolder'] = root_folder + 'Elastix/Affine_BF/' + exp_prefix + '/Affine/'
            address['MovedImAffine'] = address['AffineFolder'] + 'result.0' + ext
            address['MovedTorsoAffine'] = address['AffineFolder'] + 'MovedTorsoAffine' + ext
            address['reg_AffineOutputPoints'] = root_folder + 'Elastix/LungExp/' + exp_prefix + '/OAfterS/outputpoints.txt'

    if data == 'DIR-Lab_4D':
        if requested_address in ['rawFolder']:
            address['originalImNonIsotropicFolder'] = s['dataFolder'] + 'DIR-Lab/4DCT/mha/' + exp_prefix + '/'
            address['originalImNonIsotropic'] = address['originalImNonIsotropicFolder']+'case'+str(cn)+'_' + \
                type_im_name + ext
        elif requested_address in ['originalFolder', 'originalIm', 'originalImRaw', 'originalMask', 'originalMaskRaw',
                                   'originalTorso', 'originalTorsoRaw', 'DilatedLandmarksIm', 'DilatedLandmarksImNonIsotropic']:
            name_dic['originalIm'] = 'case' + str(cn) + '_' + type_im_name + '_RS1'
            name_dic['originalImRaw'] = 'case' + str(cn) + '_' + type_im_name
            name_dic['originalMask'] = 'Lung_Filled/case' + str(cn) + '_' + type_im_name + '_Lung_Filled_RS1'
            name_dic['originalMaskRaw'] = 'Lung_Filled/case' + str(cn) + '_' + type_im_name + '_Lung_Filled'
            name_dic['originalTorso'] = 'Torso/case' + str(cn) + '_' + type_im_name + '_Torso_RS1'
            name_dic['originalTorsoRaw'] = 'Torso/case' + str(cn) + '_' + type_im_name + '_Torso'
            name_dic['DilatedLandmarksIm'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name + '_Landmarks_RS1'
            name_dic['DilatedLandmarksImNonIsotropic'] = 'dilatedLandmarks' + str(cn) + '_' + type_im_name
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address['originalFolder'] = s['dataFolder'] + 'DIR-Lab/4DCT/mha/' + exp_prefix + '/'
            if requested_address != 'originalFolder':
                address[requested_address] = address['originalFolder'] + name_dic[requested_address] + ext

        elif requested_address in ['originalLandmarkFolder', 'LandmarkIndex', 'LandmarkIndex_tr', 'LandmarkIndex_elx',
                                   'LandmarkPoint_tr', 'LandmarkPoint_elx']:
            address['originalLandmarkFolder'] = s['dataFolder'] + 'DIR-Lab/4DCT/points/' + exp_prefix + '/'
            if ((pair_info[0]['type_im'] == 0 and pair_info[1]['type_im'] == 5) or
                (pair_info[0]['type_im'] == 5 and pair_info[1]['type_im'] == 1)) and \
                    pair_info[0]['cn'] <= 5 and pair_info[1]['cn'] <= 5:
                address['LandmarkIndex'] = address['originalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz.txt'
                address['LandmarkIndex_tr'] = address['originalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_tr.txt'
                address['LandmarkIndex_elx'] = address['originalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_xyz_elx.txt'
                address['LandmarkPoint_tr'] = address['originalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_tr.txt'
                address['LandmarkPoint_elx'] = address['originalLandmarkFolder']+'case'+str(cn)+'_300_'+type_im_name+'_world_elx.txt'
            else:
                address['LandmarkIndex'] = address['originalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz.txt'
                address['LandmarkIndex_tr'] = address['originalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_tr.txt'
                address['LandmarkIndex_elx'] = address['originalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_xyz_elx.txt'
                address['LandmarkPoint_tr'] = address['originalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_tr.txt'
                address['LandmarkPoint_elx'] = address['originalLandmarkFolder']+'case'+str(cn)+'_4D-75_'+type_im_name+'_world_elx.txt'

        elif requested_address in ['ParameterFolder']:
            address['ParameterFolder'] = root_folder + 'Elastix/Registration/Parameter/'

        elif requested_address in ['AffineFolder', 'AffineOutputTransform', 'MovedImAffine', 'MovedTorsoAffine', 'reg_AffineOutputPoints']:
            subfolder_affine = pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_' +\
                               pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'/'
            address['AffineFolder'] = root_folder + 'Elastix/Registration/Affine/' + subfolder_affine
            address['MovedImAffine'] = address['AffineFolder'] + 'result.0' + ext
            address['MovedTorsoAffine'] = address['AffineFolder'] + 'MovedTorsoAffine' + ext
            address['reg_AffineOutputPoints'] = address['AffineFolder'] + 'outputpoints.txt'

    if requested_address in ['nextFolder', 'nextIm', 'nextMask', 'nextTorso', 'nextDVF', 'nextBSplineTransform', 'nextBSplineTransformIm']:
        address['nextFolder'] = root_folder+'Elastix/'+data+'/'+deform_exp+'/'+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if print_mode:
            address['nextFolder'] = data+'/'+deform_exp+'/'+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth0'+'/DNext'+str(dsmooth)+'/'
        if requested_address in ['nextBSplineTransform']:
            address[requested_address] = address['nextFolder'] + 'nextBSplineTransform.tfm'
        if requested_address in ['nextIm', 'nextMask', 'nextTorso', 'nextDVF', 'nextBSplineTransformIm']:
            name_dic[requested_address] = requested_address
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address[requested_address] = address['nextFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['DsmoothFolder', 'DFolder', 'deformedIm', 'deformedDVF', 'deformedArea', 'deformedTorso',
                               'DVF_histogram', 'Jac', 'Jac_histogram', 'ImCanny', 'BSplineTransform', 'BSplineTransformIm']:
        dsmooth_mod = dsmooth % len(s['deform_exp'][deform_exp]['deformMethods'])
        deform_number = get_deform_number_from_dsmooth(s, dsmooth, deform_exp=deform_exp)
        address['DsmoothFolder'] = root_folder+'Elastix/'+data+'/'+deform_exp+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
        address['DFolder'] = address['DsmoothFolder']+s['deform_exp'][deform_exp]['deformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'
        if print_mode:
            address['DsmoothFolder'] = data+'/'+deform_exp+'/'+type_im_name+'/'+exp_prefix+'/Dsmooth'+str(dsmooth)+'/'
            address['DFolder'] = address['DsmoothFolder']+s['deform_exp'][deform_exp]['deformMethods'][dsmooth_mod]+'_'+'D'+str(deform_number)+'/'

        if requested_address in ['BSplineTransform']:
            address[requested_address] = address['DFolder'] + 'BSplineTransform.tfm'
        if requested_address in ['deformedIm', 'deformedDVF', 'deformedArea', 'deformedTorso', 'ImCanny', 'BSplineTransformIm']:
            name_dic['deformedIm'] = 'DeformedImage'
            # name_dic['deformedDVF'] = 'DeformedDVF'
            name_dic['deformedArea'] = 'DeformedArea'
            name_dic['deformedTorso'] = 'DeformedTorso'
            name_dic['deformedDVF'] = 'DeformedDVF' + '_pad' + str(dvf_pad)
            name_dic['ImCanny'] = 'canny'+str(s['deform_exp'][deform_exp]['onEdge-lowerThreshold'])+'_'+str(s['deform_exp'][deform_exp]['onEdge-upperThreshold'])
            name_dic['BSplineTransformIm'] = 'BSplineTransformIm'
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            address[requested_address] = address['DFolder'] + name_dic[requested_address] + ext

        if requested_address in ['DVF_histogram', 'Jac', 'Jac_histogram']:
            name_dic[requested_address] = requested_address + '_pad' + str(dvf_pad)
            if stage > 1:
                name_dic[requested_address] = name_dic[requested_address] + '_s' + str(stage)
            if requested_address in ['DVF_histogram', 'Jac_histogram']:
                address[requested_address] = address['DFolder'] + name_dic[requested_address] + '.png'
            elif requested_address in ['Jac']:
                address[requested_address] = address['DFolder'] + name_dic[requested_address] + ext

    elif requested_address in ['IShuffledFolder', 'IShuffledName', 'IShuffled']:
        if requested_address in ['IShuffledFolder', 'IShuffled']:
            ishuffled_root_folder = ''
            for my_dict in s['DataExpDict']:
                ishuffled_root_folder = ishuffled_root_folder+my_dict['data']+'_'+my_dict['deform_exp']+'_'
            address['IshuffledRootFolder'] = root_folder+'Elastix/IShuffled_'+ishuffled_root_folder[:-1]+'/'
            address['IShuffledFolder'] = address['IshuffledRootFolder'] + train_mode + '_'
            for my_dict in s['DataExpDict']:
                address['IShuffledFolder'] = address['IShuffledFolder']+my_dict['data']+'_'+my_dict['deform_exp']+'_'
                for dsmooth_ish in my_dict[train_mode + 'DSmoothList']:
                    address['IShuffledFolder'] = address['IShuffledFolder']+str(dsmooth_ish)+'_'
            address['IShuffledFolder'] = address['IShuffledFolder'][:-1]+'/S_'+str(stage)+'/'

        if requested_address in ['IShuffledName', 'IShuffled']:
            address['IShuffledName'] = train_mode + '_class'
            for str_i in s['classBalanced']:
                address['IShuffledName'] = address['IShuffledName'] + '{:.1f}_'.format(str_i)
            address['IShuffledName'] = address['IShuffledName'] + 'Margin' + str(s['Margin']) + '_Z' + str(dvf_pad) + '_numberOfImagesPerChunk' + str(
                number_of_images_per_chunk) + '_samplesPerImage' + str(samples_per_image) + '_semiEpoch' + str(semi_epoch) + '_chunk' + str(chunk) + '.npy'
        if requested_address in ['IShuffled']:
            address['IShuffled'] = address['IShuffledFolder'] + address['IShuffledName']

    elif requested_address in ['IndexFolder', 'IClassFolder', 'IClass', 'IClassName']:
        address['IndexFolder'] = root_folder + 'Elastix/' + data + '/' + deform_exp + '/' + 'Index' + '_S' + str(stage) + '/'
        address['IClassFolder'] = address['IndexFolder'] + 'IClass/'
        if requested_address in ['IClass', 'IClassName']:
            dsmooth_mod = dsmooth % len(s['deform_exp'][deform_exp]['deformMethods'])
            deform_number = get_deform_number_from_dsmooth(s, dsmooth, deform_exp=deform_exp)
            class_balanced_plus_zero = np.r_[np.array([0]), s['classBalanced']]
            address['IClassName'] = deform_exp+'_'+type_im_name+'_cn'+str(cn)+'_Dsmooth'+str(dsmooth)+'_'+s['deform_exp'][deform_exp]['deformMethods'][dsmooth_mod] +\
                '_'+'D'+str(deform_number)+'_M'+str(s['Margin'])+'_Z'+str(dvf_pad)+'_Torso'+str(int(s['torsoMask']))+'_c'+'{:.1f}_'.format(
                class_balanced_plus_zero[c])+'{:.1f}'.format(class_balanced_plus_zero[c + 1])+'.npy'
            address['IClass'] = address['IClassFolder'] + address['IClassName']

    training_log_list = ['Model_folder', 'summary_train', 'summary_test', 'log_file', 'Plots_folder', 'saved_model', 'saved_model_with_step', 'plot_fig', 'log_folder']
    real_pair_log_list = ['result_folder', 'result_step_folder', 'full_reg_folder', 'dvf_s0', 'dvf_s_up', 'moved_image', 'landmarks_file']
    if requested_address in training_log_list+real_pair_log_list:
        address['log_folder'] = s['log_root_folder'] + current_experiment + '/'
        if step is None:
            step = load_global_step_from_current_experiment(current_experiment)
        if requested_address in training_log_list:
            address['Model_folder'] = address['log_folder'] + 'train/Model/'
            address['summary_train'] = address['log_folder'] + 'train/'
            address['summary_test'] = address['log_folder'] + 'test/'
            address['log_file'] = address['Model_folder'] + 'log.txt'
            address['Plots_folder'] = address['Model_folder'] + 'Plots/'
            address['saved_model'] = address['Model_folder'] + 'Saved/RegNet3DModel.ckpt'
            address['saved_model_with_step'] = address['saved_model'] + '-' + step
            address['plot_fig'] = address['Model_folder']+'Plots/y_'+str(plot_mode)+'_itr'+str(plot_itr)+'_dir'+str(plot_i)+'.png'

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

            address['landmarks_file'] = address['result_step_folder'] + current_experiment + '-' + str(step) + '.pkl'

            if requested_address in ['full_reg_folder', 'dvf_s0', 'dvf_s_up', 'moved_image']:
                address['full_reg_folder'] = address['result_step_folder'] +\
                                             pair_info[0]['data']+'_cn'+str(pair_info[0]['cn'])+'_type_im'+str(pair_info[0]['type_im'])+'_' +\
                                             pair_info[1]['data']+'_cn'+str(pair_info[1]['cn'])+'_type_im'+str(pair_info[1]['type_im'])+'/'
                address['dvf_s0'] = address['full_reg_folder'] + 'DVF_S0' + ext
                address['dvf_s_up'] = address['full_reg_folder'] + 'DVF_S' + str(stage) + '_up' + ext
                address['moved_image'] = address['full_reg_folder'] + 'MovedImage_S' + str(stage) + ext

    return address[requested_address]


def get_deform_number_from_dsmooth(setting, dsmooth, deform_exp=None):
    if deform_exp is None:
        deform_exp = setting['DeformExpList'][0]
    deform_methods = setting['deform_exp'][deform_exp]['deformMethods']
    dsmooth_mod = dsmooth % len(deform_methods)
    selected_deform_method = deform_methods[dsmooth_mod]
    deform_method_indices = (np.where(np.array(deform_methods) == selected_deform_method))[0]
    deform_number = np.where(deform_method_indices == dsmooth_mod)[0][0]
    return int(deform_number)


def load_network_setting(setting, network_name):
    if network_name in ['decimation3', 'decimation4']:
        setting['NetworkDesign'] = network_name
        setting['NetworkInputSize'] = 77
        setting['NetworkOutputSize'] = 13
    elif network_name in['crop3_connection', 'crop4_connection']:
        setting['NetworkDesign'] = network_name
        setting['NetworkInputSize'] = 50
        setting['NetworkOutputSize'] = 10
    return setting


def get_im_info_list_from_train_mode(setting, train_mode, load_mode='Single'):
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
            for type_im in data_dict['TypeImList']:
                for cn in data_dict['CNList']:
                    for dsmooth in data_dict['DSmoothList']:
                        deform_methods = copy.deepcopy(setting['deform_exp'][data_dict['deform_exp']]['deformMethods'])
                        deform_method = deform_methods[dsmooth % len(deform_methods)]
                        deform_number = get_deform_number_from_dsmooth(setting, dsmooth, deform_exp=data_dict['deform_exp'])
                        im_info_list.append({'data': data_dict['data'], 'deform_exp': data_dict['deform_exp'], 'type_im': type_im, 'cn': cn,
                                             'dsmooth': dsmooth, 'deform_method': deform_method, 'deform_number': deform_number})
    elif load_mode == 'Pair':
        for data_dict in clean_data_exp_dict:
            for cn in data_dict['CNList']:
                for pair in data_dict['PairList']:
                    im_info_list.append([{'data': data_dict['data'], 'type_im': copy.copy(pair[0]), 'cn': cn},
                                         {'data': data_dict['data'], 'type_im': copy.copy(pair[1]), 'cn': cn}])
    else:
        raise ValueError("load_mode should be in ['Single', 'Pair'], but it is set to"+train_mode)
    return im_info_list


def load_class_balanced(setting):
    max_deform = 0
    for deform_exp in setting['DeformExpList']:
        max_deform = max(max_deform, max(setting['deform_exp'][deform_exp]['MaxDeform']))
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


def check_setting(setting):
    for deform_exp in setting['DeformExpList']:
        if setting['classBalanced'][-1] > max(setting['deform_exp'][deform_exp]['MaxDeform']):
            raise ValueError("setting['ClassBalanced'][-1] = {} should be smaller or equal to max(setting['MaxDeform']) = {}  ".format(
                setting['classBalanced'][-1], max(setting['MaxDeform'])))
    if (setting['R']-setting['Ry']) < 0:
        raise ValueError("setting['R'] = {} should be greater or equal to setting['Ry'] = {}  ".format(
            setting['R'], setting['Ry']))
    if 'Randomness' in setting:
        if not setting['Randomness']:
            logging.warning('----!!!!!! setting["Randomness"] is set to False, network might not be trained correctly')
    if not setting['ParallelProcessing']:
        logging.warning('----!!!!!! setting["ParallelProcessing"] is set to False, running might be very slow')
    if not setting['ParallelImageGeneration']:
        logging.warning('----!!!!!! setting["ParallelImageGeneration"] is set to False, running might be very slow')

    if setting['DetailedSummary']:
        logging.warning('----!!!!!! setting["DetailedSummary"] is set to True, running might be very slow')

    im_list_info = get_im_info_list_from_train_mode(setting, train_mode='Training')
    number_of_images_last_chunk = len(im_list_info) % setting['NetworkTraining']['NumberOfImagesPerChunk']
    logging.warning('number of images in the last chunk='+str(number_of_images_last_chunk))
    if 0 < number_of_images_last_chunk < 10:
        logging.warning('----!!!! Small number of images are left for the last chunk. Total number of images in the Training='+str(len(im_list_info)) +
                        ' and NumberOfImagesPerChunk='+str(setting['NetworkTraining']['NumberOfImagesPerChunk']) +
                        ', number of images in the last chunk='+str(number_of_images_last_chunk))


def write_setting(setting, setting_address=None):
    if setting_address is None:
        setting_folder = address_generator(setting, 'Model_folder')
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
