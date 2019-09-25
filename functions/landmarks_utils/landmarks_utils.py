import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import dill
from scipy.stats import wilcoxon
import SimpleITK as sitk
import time
import xlsxwriter
import functions.setting.setting_utils as su
from .image import landmark_info


def calculate_write_landmark(setting, pair_info, overwrite_landmarks=False, overwrite_landmarks_hard=False,
                             overwrite_bspline_dvf=False, overwrite_jac=False):
    """
    Add the following information:     [setting, 'pair_info', 'FixedLandmarksWorld', 'MovingLandmarksWorld', 'FixedAfterAffineLandmarksWorld'
                                        'DVFAffine', 'DVF_nonrigidGroundTruth', 'FixedLandmarksIndex']
    if RegNet registration available:   'DVFRegNet'
    if BSpline registration available:  'DVF_BSpline'
    if Jac available:                   'Jac_NumberOfNegativeJacVoxels', 'Jac_MaskSize', 'Jac_Var'

    :param setting:
    :param pair_info:
    :param overwrite_landmarks:
    :param overwrite_bspline_dvf:
    :param overwrite_jac:
    :return:
    """
    time_before = time.time()
    stage_list = setting['ImagePyramidSchedule']
    im_info_fixed = copy.deepcopy(pair_info[0])
    im_info_fixed['stage'] = 1
    landmark_address = su.address_generator(setting, 'landmarks_file', stage_list=stage_list)
    if os.path.isfile(landmark_address):
        with open(landmark_address, 'rb') as f:
            landmark = dill.load(f)
    else:
        landmark = []
        result_landmarks_folder = su.address_generator(setting, 'result_landmarks_folder', stage_list=stage_list)
        if not os.path.isdir(result_landmarks_folder):
            os.makedirs(result_landmarks_folder)

    if setting['current_experiment'] == 'elx_registration':
        jac_name = 'DVFBSpline_Jac'
    else:
        jac_name = 'dvf_s0_jac'
    dvf_bspline_address = su.address_generator(setting, 'DVFBSpline', pair_info=pair_info, **im_info_fixed)
    dvf_s0_address = su.address_generator(setting, 'dvf_s0', pair_info=pair_info, stage_list=stage_list)
    jac_address = su.address_generator(setting, jac_name, pair_info=pair_info, stage_list=stage_list)
    read_regnet_dvf = True
    read_bspline_dvf = True
    read_jac = True
    if not os.path.isfile(dvf_bspline_address):
        read_bspline_dvf = False
    if not os.path.isfile(dvf_s0_address):
        read_regnet_dvf = False
    if not os.path.isfile(jac_address):
        read_jac = False

    pair_info_text = 'stage_list={}'.format(stage_list) + ' Fixed: ' + pair_info[0]['data'] + \
                     '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '  Moving:' + \
                     pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(pair_info[1]['cn'], pair_info[1]['type_im'])
    ind_find_list = [compare_pair_info_dict(pair_info, landmark_i['pair_info'], compare_keys=['data', 'cn', 'type_im'])
                     for landmark_i in landmark]
    ind_find = None
    if any(ind_find_list):
        ind_find = ind_find_list.index(True)
        if not overwrite_landmarks and 'DVFRegNet' in landmark[ind_find]:
            read_regnet_dvf = False
        else:
            logging.debug('Landmark overwriting ' + pair_info_text)

        if not overwrite_bspline_dvf and 'DVF_BSpline' in landmark[ind_find]:
            read_bspline_dvf = False
        else:
            logging.debug('DVF Bspline overwriting ' + pair_info_text)

        if not overwrite_jac and 'Jac_NumberOfNegativeJacVoxels' in landmark[ind_find]:
            read_jac = False
        else:
            logging.debug('Jac overwriting ' + pair_info_text)

    # if not read_regnet_dvf:
    #     logging.debug('Landmark skipping ' + pair_info_text)
    # if not read_bspline_dvf:
    #     logging.debug('DVF Bspline skipping ' + pair_info_text)
    # if not read_jac:
    #     logging.debug('Jac skipping ' + pair_info_text)

    # if overwrite_landmarks_hard:
    #     del landmark[ind_find]
    #     ind_find = None
        # delete the key and start from the begining

    if ind_find is None:
        landmark_dict = landmark_info(setting, pair_info)
        landmark_dict['network_dict'] = copy.deepcopy(setting['network_dict'])
        landmark.append(landmark_dict)
        ind_find = -1

    if overwrite_landmarks_hard:
        # temp
        landmark_dict = landmark_info(setting, pair_info)
        keys_to_copy = ['FixedLandmarksWorld', 'MovingLandmarksWorld', 'FixedAfterAffineLandmarksWorld',
                        'DVFAffine', 'DVF_nonrigidGroundTruth', 'FixedLandmarksIndex']
        for key in keys_to_copy:
            landmark[ind_find][key] = copy.deepcopy(landmark_dict[key])

    if read_regnet_dvf:
        dvf_s0 = sitk.GetArrayFromImage(sitk.ReadImage(dvf_s0_address))
        fixed_landmarks_index = landmark[ind_find]['FixedLandmarksIndex']
        # be careful about the order. 'FixedLandmarksIndex': xyz order
        landmark[ind_find]['DVFRegNet'] = np.stack([dvf_s0[fixed_landmarks_index[i, 2],
                                                           fixed_landmarks_index[i, 1],
                                                           fixed_landmarks_index[i, 0]]
                                                    for i in range(len(fixed_landmarks_index))])

    if read_bspline_dvf:
        dvf_bspline = sitk.GetArrayFromImage(sitk.ReadImage(dvf_bspline_address))
        fixed_landmarks_index = landmark[ind_find]['FixedLandmarksIndex']
        landmark[ind_find]['DVF_BSpline'] = np.stack([dvf_bspline[fixed_landmarks_index[i, 2],
                                                                  fixed_landmarks_index[i, 1],
                                                                  fixed_landmarks_index[i, 0]]
                                                      for i in range(len(fixed_landmarks_index))])

    if read_jac:
        mask_to_zero = setting['network_dict']['stage'+str(setting['ImagePyramidSchedule'][-1])]['MaskToZero']
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(su.address_generator(setting, mask_to_zero, **im_info_fixed)))
        jac = sitk.GetArrayFromImage(sitk.ReadImage(jac_address))
        jac = jac * fixed_mask
        # you can remove the following if statement later.
        if 'Jac' in landmark[ind_find].keys():
            landmark[ind_find].pop('Jac', None)
        landmark[ind_find]['Jac_NumberOfNegativeJacVoxels'] = np.sum(jac[fixed_mask > 0] < 0)
        landmark[ind_find]['Jac_MaskSize'] = np.sum(fixed_mask > 0)
        landmark[ind_find]['Jac_Var'] = np.var(jac[fixed_mask > 0])

    if read_regnet_dvf or read_bspline_dvf or read_jac or overwrite_landmarks_hard or ind_find == -1:
        with open(landmark_address, 'wb') as f:
            dill.dump(landmark, f)
    time_after = time.time()
    logging.debug('Landmark ' + pair_info_text + ' is done in {:.2f}s '.format(time_after - time_before))

    return landmark


def load_landmarks(setting, pair_info_list, experiment_list_new):
    """
    It loads the landmarks file from different experiments. each experiment has a separate landmarks file. It also calculates the 'Error' and 'TRE' for each pair
    For each pair it creates a dictionary with two keys:
        'pair_info': a copy of pair_info
        'landmark_info: a copy of all keys in that pair +  'Error' + 'TRE'

    :param setting:
    :param pair_info_list:
    :param experiment_list_new:
    :return: landmarks_dict: a dictionary of different experiments:
                             structure: landmarks_dict['experiment1'][list of all pairs]['pair_info', 'landmark_info']
    """
    landmarks_dict = dict()
    for exp_dict in experiment_list_new:
        exp_pure = exp_dict['experiment']
        stage_list = exp_dict['stage_list']
        exp = copy.copy(exp_pure)
        for stage_str in exp_dict['stage_list']:
            exp = exp + '_' + 'S' + str(stage_str)
        exp_folder = exp_pure.split('-')[0]
        if exp_pure in ['Affine', 'BSpline']:
            landmark_address = su.address_generator(setting, 'landmarks_file', current_experiment='elx_registration', stage_list=stage_list)
        else:
            landmark_address = su.address_generator(setting, 'landmarks_file', current_experiment=exp_folder, stage_list=stage_list)
        with open(landmark_address, 'rb') as f:
            landmarks_load = dill.load(f)
        landmarks_dict[exp] = []
        for pair_info in pair_info_list:
            landmark_pair = dict()
            pair_info_text = exp + ' Fixed: ' + pair_info[0]['data'] + \
                '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '  Moving:' + \
                pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(pair_info[1]['cn'], pair_info[1]['type_im'])
            print('loading ' + pair_info_text)
            ind_find_list = [compare_pair_info_dict(pair_info, landmark_i['pair_info'], compare_keys=['data', 'cn', 'type_im'])
                             for landmark_i in landmarks_load]
            if any(ind_find_list):
                ind_find = ind_find_list.index(True)
                landmark_pair = copy.deepcopy(landmarks_load[ind_find])
            else:
                print('landmark not found in experiment:' + pair_info_text)

            if exp_pure == 'Affine':
                estimated_dvf = np.zeros(np.shape(landmark_pair['MovingLandmarksWorld']))
            elif exp_pure == 'BSpline':
                estimated_dvf = landmark_pair['DVF_BSpline']
            else:
                try:
                    estimated_dvf = landmark_pair['DVFRegNet']
                except:
                    logging.warning('DVFRegNet not found ' + pair_info_text)
            landmark_pair['Error'] = np.array([landmark_pair['MovingLandmarksWorld'][i, :] -
                                               landmark_pair['FixedAfterAffineLandmarksWorld'][i, :] -
                                               estimated_dvf[i, :]
                                               for i in range(np.shape(landmark_pair['MovingLandmarksWorld'])[0])])
            landmark_pair['TRE'] = np.array([np.linalg.norm(landmark_pair['Error'][i, :])
                                             for i in range(np.shape(landmark_pair['MovingLandmarksWorld'])[0])])

            landmark_all_info = {'pair_info': copy.deepcopy(pair_info),
                                 'landmark_info': landmark_pair}

            landmarks_dict[exp].append(landmark_all_info)
    return landmarks_dict


def table_box_plot(setting, landmarks, compare_list=None, fig_measure_list=None, plot_per_pair=False, fig_ext='.png',
                   plot_folder=None, paper_table=None, naming_strategy=None, jacobian=False, multi_jac_exp_list=None):
    """
    merge the landmarks from different cases
    :param setting:
    :param landmarks:
    :param compare_list:
    :param
    :param plot_folder: one of the experiment to save all plots in that directory. The plot folder should consider also the stages
    for example: my_experiment_S4_S2_S1. if None, the last experiment will be chosen as the plot folder
    :param naming_strategy: None
                            'Fancy'
                            'Clean'
    :param paper_table: 'SPREAD', 'DIR-Lab'
    :return:
    """
    # comment later:
    # exp_txt = '2020_multistage_crop45_K_Resp_more_itr_S4_S2_S1'
    # for i in range(10):
    #     error_matrix = landmarks[exp_txt][i]['landmark_info']['Error']
    #     file_name = 'E:/PHD/R/case'+str(i+1)+'.txt'
    #     np.savetxt(file_name, error_matrix, fmt='%8.4f',)

    if multi_jac_exp_list is None:
        if paper_table == 'SPREAD-Multi':
            multi_jac_exp_list = ['2020_multistage_crop45_K_NoResp_SingleOnly_more_itr_S4_S2_S1',
                                  'BSpline_S4_S2_S1']

            multi_jac_exp_list = [
                # '2020_multistage_crop45_K_NoResp_more_itr_S4_S2_S1',
                                  # 'BSpline_S4_S2_S1'
                                  ]
        if paper_table == 'DIR-Lab-Multi':
            multi_jac_exp_list = ['2020_multistage_crop45_K_Resp_more_itr_S4_S2_S1']

    if compare_list is None:
        compare_list = ['Affine', 'BSpline']
    if fig_measure_list is None:
        fig_measure_list = ['TRE']

    landmarks_merged = dict()
    if plot_folder is None:
        plot_key = list(landmarks.items())[-1][0]
    else:
        plot_key = plot_folder
    split_plot_folder = plot_key.split('_S')
    stage_list = []
    plot_folder_key_pure = split_plot_folder[0]
    for split_str in split_plot_folder:
        if len(split_str) == 1:
            stage_list.append(int(split_str))
    result_folder = su.address_generator(setting, 'result_detail_folder',
                                         current_experiment=plot_folder_key_pure,
                                         stage_list=stage_list,
                                         pair_info=landmarks[plot_key][0]['pair_info'])
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    xlsx_address = result_folder + 'results.xlsx'
    # if os.path.isfile(xlsx_address):
    #     raise ValueError(xlsx_address + 'already exists cannot overwrite')
    workbook = xlsxwriter.Workbook(xlsx_address)
    worksheet = workbook.add_worksheet()
    line = 0
    header = {'exp': 0, 'TRE_Mean': 1, 'TRE_STD': 2, 'TRE_Median': 3, 'MAE0_Mean': 4, 'MAE0_STD': 5,
              'MAE1_Mean': 6, 'MAE1_STD': 7, 'MAE2_Mean': 8, 'MAE2_STD': 9, 'Jac_FoldingPercentage': 10,
              'Jac_STD': 11, 'Error0_Mean': 12, 'Error0_STD': 13, 'Error1_Mean': 14, 'Error1_STD': 15,
              'Error2_Mean': 16, 'Error2_STD': 17, }

    for key in header.keys():
        worksheet.write(line, header[key], key)
    num_exp = len(landmarks.keys())
    for exp_i, exp in enumerate(landmarks.keys()):
        landmarks_merged[exp] = {'TRE': np.empty([0]),
                                 'Error': np.empty([0, 3]),
                                 'CleanName': su.clean_exp_name(exp),
                                 'FancyName': su.fancy_exp_name(exp),
                                 'Jac_NumberOfNegativeJacVoxels': 0,
                                 'Jac_MaskSize': 0,
                                 'Jac_Var': 0,
                                 'Jac_STD_List': [],
                                 'Jac_FoldingPercentage_List': [],
                                 }

        num_pair = len(landmarks[exp])
        for pair_i, landmark_pair in enumerate(landmarks[exp]):
            pair_info = landmark_pair['pair_info']
            pair_info_text = landmarks_merged[exp]['CleanName'] + '_Fixed_' + pair_info[0]['data'] + \
                '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '_Moving_' + \
                pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(pair_info[1]['cn'], pair_info[1]['type_im'])
            landmarks_merged[exp]['TRE'] = np.append(landmarks_merged[exp]['TRE'], landmark_pair['landmark_info']['TRE'])
            landmarks_merged[exp]['Error'] = np.vstack((landmarks_merged[exp]['Error'], landmark_pair['landmark_info']['Error']))
            if jacobian:
                if paper_table is not None:
                    if exp in multi_jac_exp_list:
                        landmarks_merged[exp]['Jac_NumberOfNegativeJacVoxels'] += landmark_pair['landmark_info']['Jac_NumberOfNegativeJacVoxels']
                        landmarks_merged[exp]['Jac_MaskSize'] += landmark_pair['landmark_info']['Jac_MaskSize']
                        landmarks_merged[exp]['Jac_Var'] += landmark_pair['landmark_info']['Jac_Var']*1/len(landmarks[exp]) # on line averaging for Var
                        landmarks_merged[exp]['Jac_STD_List'].append(np.sqrt(landmark_pair['landmark_info']['Jac_Var']))
                        landmarks_merged[exp]['Jac_FoldingPercentage_List'].append(
                            landmark_pair['landmark_info']['Jac_NumberOfNegativeJacVoxels']/landmark_pair['landmark_info']['Jac_MaskSize']*100)

            measure = calculate_measure(landmark_pair['landmark_info'])
            measure['exp'] = pair_info_text
            if plot_per_pair:
                print_latex(measure)
            line = exp_i + pair_i * (num_exp + 1) + 1
            for key in header.keys():
                if key in measure.keys():
                    worksheet.write(line, header[key], measure[key])
                    landmark_pair['landmark_info'][key] = measure[key]

        measure_merged = calculate_measure(landmarks_merged[exp])
        if naming_strategy == 'Clean':
            measure_merged['exp'] = su.clean_exp_name(exp)
        elif naming_strategy == 'Fancy':
            measure_merged['exp'] = su.fancy_exp_name(exp)
        else:
            measure_merged['exp'] = exp
        print_latex(measure_merged)
        line = exp_i + num_pair * (num_exp + 1) + 2
        for key in header.keys():
            if key in measure.keys():
                if key in header.keys() and key in measure_merged.keys():
                    worksheet.write(line, header[key], measure_merged[key])
    workbook.close()

    if paper_table == 'SPREAD':
        print_latex_spread(landmarks, landmarks_merged, plot_key)
    if paper_table == 'DIR-Lab':
        print_latex_dir_lab_4d(landmarks, landmarks_merged, plot_key)
    if paper_table == 'SPREAD-Multi':
        print_latex_spread_multiple(landmarks, landmarks_merged, multi_jac_exp_list)
    if paper_table == 'DIR-Lab-Multi':
        print_latex_dir_lab_4d_multiple(landmarks, landmarks_merged, multi_jac_exp_list)
    for measure in fig_measure_list:
        if plot_per_pair:
            for pair_i in range(len(landmarks[next(iter(landmarks))])):
                fig, ax = plt.subplots(figsize=(15, 8))
                bplot1 = plt.boxplot([landmarks[exp][pair_i]['landmark_info'][measure] for exp in landmarks.keys()],
                                     patch_artist=True, notch=True)
                title_name = landmarks[next(iter(landmarks))][pair_i]['landmark_info']['exp']
                plt.title(title_name)
                plt.draw()
                plt.savefig(result_folder+measure+'_'+title_name+fig_ext)
                plt.close()

        fig, ax = plt.subplots(figsize=(15, 8))
        bplot1 = plt.boxplot([landmarks_merged[exp][measure] for exp in landmarks_merged.keys()],
                             patch_artist=True, notch=True)
        title_name = measure + '_Merged'
        plt.title(title_name)
        plt.savefig(result_folder + measure + '_' + title_name + fig_ext)
        plt.draw()
        plt.close()


def calculate_measure(input_dict, calculate_tre=True, calculate_mae=True, calculate_error=True, calculate_jac=True):
    measure = dict()
    if calculate_tre:
        measure['TRE_Mean'] = np.mean(input_dict['TRE'])
        measure['TRE_STD'] = np.std(input_dict['TRE'])
        measure['TRE_Median'] = np.median(input_dict['TRE'])
    for dim in range(3):
        if calculate_error:
            measure['Error'+str(dim)+'_Mean'] = np.mean(input_dict['Error'][:, dim])
            measure['Error'+str(dim)+'_STD'] = np.std(input_dict['Error'][:, dim])
        if calculate_mae:
            measure['MAE'+str(dim)+'_Mean'] = np.mean(np.abs(input_dict['Error'][:, dim]))
            measure['MAE'+str(dim)+'_STD'] = np.std(np.abs(input_dict['Error'][:, dim]))
    if calculate_jac:
        if 'Jac_MaskSize' in input_dict.keys():
            if input_dict['Jac_MaskSize'] != 0:
                measure['Jac_FoldingPercentage'] = input_dict['Jac_NumberOfNegativeJacVoxels'] / input_dict['Jac_MaskSize'] * 100
                measure['Jac_STD'] = np.sqrt(input_dict['Jac_Var'])
                if 'Jac_STD_List' in input_dict.keys():
                    measure['Jac_STD_STD'] = np.std(input_dict['Jac_STD_List'])
                if 'Jac_FoldingPercentage_List' in input_dict.keys():
                    measure['Jac_FoldingPercentage_STD'] = np.std(input_dict['Jac_FoldingPercentage_List'])

    return measure


def print_latex(measure):
    latex_str = measure['exp'] + ' &Single+Mixed'
    latex_str += '&\scriptsize${:.2f}'.format(measure['TRE_Mean'])+'{\pm}'+'{:.2f}$'.format(measure['TRE_STD'])
    for dim in range(3):
        latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure['MAE'+str(dim)+'_Mean'])+'{\pm}'+\
                     '{:.2f}$'.format(measure['MAE'+str(dim)+'_STD'])
    if 'Jac_FoldingPercentage' in measure.keys():
        latex_str += ' &\scriptsize${:.2f}$'.format(measure['Jac_FoldingPercentage'])
        if 'Jac_FoldingPercentage_STD' in measure.keys():
            latex_str += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure['Jac_FoldingPercentage_STD'])
    if 'Jac_STD' in measure.keys():
        latex_str += ' &\scriptsize${:.2f}$'.format(measure['Jac_STD'])
        if 'Jac_STD_STD' in measure.keys():
            latex_str += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure['Jac_STD_STD'])
    latex_str += r'\\'
    print(latex_str)
    return 0


def print_latex_spread(landmarks, landmarks_merged, plot_folder):
    ref_experiment_list = ['Affine_S4_S2_S1', 'BSpline_S4_S2_S1']
    for pair_i, landmark_pair in enumerate(landmarks['Affine_S4_S2_S1']):
        latex_str = 'case '+ str(landmark_pair['pair_info'][0]['cn'])
        for ref_exp in ref_experiment_list:
            measure_ref = calculate_measure(landmarks[ref_exp][pair_i]['landmark_info'])
            latex_str += ' &\scriptsize${:.2f}'.format(measure_ref['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_ref['TRE_STD'])
        measure_exp = calculate_measure(landmarks[plot_folder][pair_i]['landmark_info'])
        latex_str += ' &\scriptsize${:.2f}'.format(measure_exp['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_exp['TRE_STD'])
        for dim in range(3):
            latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_exp['MAE'+str(dim)+'_Mean'])+'{\pm}'+\
                         '{:.2f}$'.format(measure_exp['MAE'+str(dim)+'_STD'])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_FoldingPercentage'])
        if 'Jac_FoldingPercentage_STD' in measure_exp.keys():
            latex_str += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure_exp['Jac_FoldingPercentage_STD'])

        latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_STD'])
        if 'Jac_STD_STD' in measure_exp.keys():
            latex_str += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure_exp['Jac_STD_STD'])

        latex_str += r'\\'
        print(latex_str)

    latex_str = 'Total'
    for ref_exp in ref_experiment_list:
        measure_ref = calculate_measure(landmarks_merged[ref_exp])
        latex_str += ' &\scriptsize${:.2f}'.format(measure_ref['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_ref['TRE_STD'])
    measure_merged_exp = calculate_measure(landmarks_merged[plot_folder])
    latex_str += ' &\scriptsize${:.2f}'.format(measure_merged_exp['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['TRE_STD'])
    for dim in range(3):
        latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_merged_exp['MAE' + str(dim) + '_Mean']) + '{\pm}' + \
                     '{:.2f}$'.format(measure_merged_exp['MAE' + str(dim) + '_STD'])
    latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage'])
    if 'Jac_FoldingPercentage_STD' in measure_merged_exp.keys():
        latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage_STD'])

    latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_STD'])
    if 'Jac_STD_STD' in measure_merged_exp.keys():
        latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_STD_STD'])
    latex_str += r'\\'
    print(latex_str)


def print_latex_spread_multiple(landmarks, landmarks_merged, multi_jac_exp_list):
    ref_experiment_list = ['Affine_S4_S2_S1', 'BSpline_S4_S2_S1']
    all_exp_list = list(landmarks.keys())
    # regnet_exp_list = [i for i in all_exp_list if i not in ref_experiment_list and i in all_exp_list]
    # exp_significance = 'BSpline_S4_S2_S1'
    exp_significance = '2020_multistage_crop45_K_NoResp_more_itr_S4_S2_S1'
    alpha = 0.05

    for pair_i, landmark_pair in enumerate(landmarks['Affine_S4_S2_S1']):
        latex_str = 'case '+ str(landmark_pair['pair_info'][0]['cn'])
        tre_mean_list = np.array([])
        tre_std_list = np.array([])
        significance_list = np.array([], dtype=np.bool)

        tre_significance_ref = landmarks[exp_significance][pair_i]['landmark_info']['TRE']
        for exp in all_exp_list:
            measure_ref = calculate_measure(landmarks[exp][pair_i]['landmark_info'])
            tre_mean_list = np.append(tre_mean_list, measure_ref['TRE_Mean'])
            tre_std_list = np.append(tre_std_list, measure_ref['TRE_STD'])
            tre_significance_exp = landmarks[exp][pair_i]['landmark_info']['TRE']
            statistic, pvalue = wilcoxon(tre_significance_ref, tre_significance_exp, zero_method='zsplit')
            # print('pvalue={:.2f}'.format(pvalue))
            if pvalue > alpha:
                # same distribution
                significance_list = np.append(significance_list, False)
            else:
                significance_list = np.append(significance_list, True)

        latex_str = latex_color_print(latex_str, tre_mean_list, tre_std_list, significance_list=significance_list)

        for multi_jac_exp in multi_jac_exp_list:
            measure_exp = calculate_measure(landmarks[multi_jac_exp][pair_i]['landmark_info'])
            jac_string = ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_FoldingPercentage'])
            if 'Jac_FoldingPercentage_STD' in measure_exp.keys():
                jac_string += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure_exp['Jac_FoldingPercentage_STD'])

            jac_string += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_STD'])
            if 'Jac_STD_STD' in measure_exp.keys():
                jac_string += '\scriptsize$'+'{\pm}'+'{:.2f}$'.format(measure_exp['Jac_STD_STD'])
            if multi_jac_exp == 'BSpline_S4_S2_S1':
                and_positions = [pos for pos, char in enumerate(latex_str) if char == '&']
                latex_str = latex_str[:and_positions[2]] + jac_string + latex_str[and_positions[2]:]
            else:
                latex_str += jac_string

        latex_str += r'\\'
        print(latex_str)

    latex_str = 'Total'
    tre_mean_list = np.array([])
    tre_std_list = np.array([])
    significance_list = np.array([], dtype=np.bool)

    tre_significance_ref = landmarks_merged[exp_significance]['TRE']
    for exp in all_exp_list:
        measure_ref = calculate_measure(landmarks_merged[exp])
        tre_mean_list = np.append(tre_mean_list, measure_ref['TRE_Mean'])
        tre_std_list = np.append(tre_std_list, measure_ref['TRE_STD'])
        tre_significance_exp = landmarks_merged[exp]['TRE']
        statistic, pvalue = wilcoxon(tre_significance_ref, tre_significance_exp, zero_method='zsplit')
        # print('pvalue={:.2f}'.format(pvalue))
        if pvalue > alpha:
            # same distribution
            significance_list = np.append(significance_list, False)
        else:
            significance_list = np.append(significance_list, True)

    latex_str = latex_color_print(latex_str, tre_mean_list, tre_std_list, significance_list=significance_list)

    # for dim in range(3):
    #     latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_merged_exp['MAE' + str(dim) + '_Mean']) + '{\pm}' + \
    #                  '{:.2f}$'.format(measure_merged_exp['MAE' + str(dim) + '_STD'])
    for multi_jac_exp in multi_jac_exp_list:
        measure_merged_exp = calculate_measure(landmarks_merged[multi_jac_exp])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage'])
        if 'Jac_FoldingPercentage_STD' in measure_merged_exp.keys():
            latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage_STD'])

        latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_STD'])
        if 'Jac_STD_STD' in measure_merged_exp.keys():
            latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_STD_STD'])
    latex_str += r'\\'
    print(latex_str)


def print_latex_dir_lab_4d(landmarks, landmarks_merged, plot_folder):
    ref_experiment_list = ['Affine_S4_S2_S1', 'Floris', 'GDL_FIRE', 'Bob', 'Koen']
    tre_state_of_arts = get_tre_state_of_arts()
    for pair_i, landmark_pair in enumerate(landmarks['Affine_S4_S2_S1']):
        cn = landmark_pair['pair_info'][0]['cn']
        latex_str = str(cn).zfill(2)
        for ref_exp in ref_experiment_list:
            if ref_exp == 'Affine_S4_S2_S1':
                measure_ref = calculate_measure(landmarks[ref_exp][pair_i]['landmark_info'])
            else:
                measure_ref = tre_state_of_arts[ref_exp]['Case'+str(cn)]
            latex_str += ' &\scriptsize${:.2f}'.format(measure_ref['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_ref['TRE_STD'])
        measure_exp = calculate_measure(landmarks[plot_folder][pair_i]['landmark_info'])
        latex_str += ' &\scriptsize${:.2f}'.format(measure_exp['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_exp['TRE_STD'])
        for dim in range(3):
            latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_exp['MAE'+str(dim)+'_Mean'])+'{\pm}'+\
                         '{:.2f}$'.format(measure_exp['MAE'+str(dim)+'_STD'])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_FoldingPercentage'])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_STD'])
        latex_str += r'\\'
        print(latex_str)

    latex_str = 'Total'
    for ref_exp in ref_experiment_list:
        if ref_exp == 'Affine_S4_S2_S1':
            measure_ref = calculate_measure(landmarks_merged[ref_exp])
        else:
            measure_ref = tre_state_of_arts[ref_exp]['Total']
        latex_str += ' &\scriptsize${:.2f}'.format(measure_ref['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_ref['TRE_STD'])
    measure_merged_exp = calculate_measure(landmarks_merged[plot_folder])
    latex_str += ' &\scriptsize${:.2f}'.format(measure_merged_exp['TRE_Mean']) + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['TRE_STD'])
    for dim in range(3):
        latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_merged_exp['MAE' + str(dim) + '_Mean']) + '{\pm}' + \
                     '{:.2f}$'.format(measure_merged_exp['MAE' + str(dim) + '_STD'])

    latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage'])
    if 'Jac_FoldingPercentage_STD' in measure_merged_exp.keys():
        latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage_STD'])
    latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_STD'])
    if 'Jac_STD_STD' in measure_merged_exp.keys():
        latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_STD_STD'])

    latex_str += r'\\'
    print(latex_str)


def print_latex_dir_lab_4d_multiple(landmarks, landmarks_merged, multi_jac_exp_list):
    all_exp_list = list(landmarks.keys())
    regnet_exp_list = [i for i in all_exp_list if i not in ['Affine_S4_S2_S1', 'BSpline_S4_S2_S1'] and i in all_exp_list]
    elx_exp_list = ['Affine_S4_S2_S1',
                    'BSpline_S4_S2_S1',
                    ]
    state_of_arts_list = ['Floris',
                           'GDL_FIRE',
                           'Bob',
                           'Koen',
                           'Koen_DIR']
    # all_ref_exp = elx_exp_list + state_of_arts_list
    all_exp_list = elx_exp_list + state_of_arts_list + regnet_exp_list
    tre_state_of_arts = get_tre_state_of_arts()
    exp_significance = 'Floris'
    alpha = 0.05

    for pair_i, landmark_pair in enumerate(landmarks['Affine_S4_S2_S1']):
        cn = landmark_pair['pair_info'][0]['cn']
        latex_str = 'case ' + str(cn).zfill(2)
        tre_mean_list = np.array([])
        tre_std_list = np.array([])
        for exp in all_exp_list:
            if exp in state_of_arts_list:
                measure_ref = tre_state_of_arts[exp]['Case' + str(cn)]
            else:
                measure_ref = calculate_measure(landmarks[exp][pair_i]['landmark_info'])
            tre_mean_list = np.append(tre_mean_list, measure_ref['TRE_Mean'])
            tre_std_list = np.append(tre_std_list, measure_ref['TRE_STD'])
        latex_str = latex_color_print(latex_str, tre_mean_list, tre_std_list)

        for multi_jac_exp in multi_jac_exp_list:
            measure_exp = calculate_measure(landmarks[multi_jac_exp][pair_i]['landmark_info'])
            latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_FoldingPercentage'])
            latex_str += ' &\scriptsize${:.2f}$'.format(measure_exp['Jac_STD'])

        latex_str += r'\\'
        latex_str = latex_str.replace('\scriptsize$100.00{\pm}100.00$', '-')
        latex_str += '\n'
        print(latex_str)

    latex_str = 'Total'
    tre_mean_list = np.array([])
    tre_std_list = np.array([])
    significance_list = np.array([])
    if exp_significance in state_of_arts_list:
        tre_significance_ref = np.array([tre_state_of_arts[exp_significance]['Case'+str(i)]['TRE_Mean'] for i in range(1, 11)])
    else:
        tre_significance_ref = landmarks_merged[exp_significance]['TRE']
    for exp in all_exp_list:
        if exp in state_of_arts_list:
            measure_ref = tre_state_of_arts[exp]['Total']
            tre_significance_exp = np.array([tre_state_of_arts[exp]['Case' + str(i)]['TRE_Mean'] for i in range(1, 11)])
        else:
            measure_ref = calculate_measure(landmarks_merged[exp])
            tre_significance_exp = landmarks_merged[exp]['TRE']

        tre_mean_list = np.append(tre_mean_list, measure_ref['TRE_Mean'])
        tre_std_list = np.append(tre_std_list, measure_ref['TRE_STD'])

        statistic, pvalue = wilcoxon(tre_significance_ref, tre_significance_exp, zero_method='zsplit')
        if pvalue > alpha:
            # same distribution
            significance_list = np.append(significance_list, False)
        else:
            significance_list = np.append(significance_list, True)

    # for regnet_exp in regnet_exp_list:
    #     measure_merged_exp = calculate_measure(landmarks_merged[regnet_exp])
    #     tre_mean_list = np.append(tre_mean_list, measure_merged_exp['TRE_Mean'])
    #     tre_std_list = np.append(tre_std_list, measure_merged_exp['TRE_STD'])

    latex_str = latex_color_print(latex_str, tre_mean_list, tre_std_list, significance_list=significance_list)

    # for dim in range(3):
    #     latex_str += ' & ' + '\scriptsize${:.2f}'.format(measure_merged_exp['MAE' + str(dim) + '_Mean']) + '{\pm}' + \
    #                  '{:.2f}$'.format(measure_merged_exp['MAE' + str(dim) + '_STD'])
    #
    for multi_jac_exp in multi_jac_exp_list:
        measure_merged_exp = calculate_measure(landmarks_merged[multi_jac_exp])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage'])
        if 'Jac_FoldingPercentage_STD' in measure_merged_exp.keys():
            latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_FoldingPercentage_STD'])
        latex_str += ' &\scriptsize${:.2f}$'.format(measure_merged_exp['Jac_STD'])
        if 'Jac_STD_STD' in measure_merged_exp.keys():
            latex_str += '\scriptsize$' + '{\pm}' + '{:.2f}$'.format(measure_merged_exp['Jac_STD_STD'])

    latex_str += r'\\'
    latex_str = latex_str.replace('\scriptsize$100.00{\pm}100.00$', '-')
    latex_str = r'\rule{0pt}{3ex}' + latex_str
    print(latex_str)


def latex_color_print(latex_str, tre_mean_list, tre_std_list, significance_list=None, scriptsize=True):
    sort_ind = np.argsort(tre_mean_list)[0:2]
    i_bold = sort_ind[0]
    i_green = sort_ind[1]
    if np.round(tre_mean_list[sort_ind[0]], decimals=2) == np.round(tre_mean_list[sort_ind[1]], decimals=2):
        tre_sum = np.round(tre_mean_list[sort_ind], decimals=2) + np.round(tre_std_list[sort_ind], decimals=2)
        if tre_sum[1] > tre_sum[0]:
            i_bold = sort_ind[1]
            i_green = sort_ind[0]

    for i in range(len(tre_mean_list)):
        write_significane = False
        if significance_list is not None:
            if significance_list[i]:
                write_significane = True

        latex_exp = r' &' + '${:.2f}'.format(tre_mean_list[i]) + '{\pm}' + '{:.2f}$'.format(tre_std_list[i])
        if i == i_bold:
            latex_exp = r' &\boldmath{' + latex_exp[2:] + '}'
        if i == i_green:
            latex_exp = r' &\tcb{' + latex_exp[2:] + '}'

        if scriptsize:
            latex_exp = ' &\scriptsize' + latex_exp[2:]

        if write_significane:
            if i == i_bold or i == i_green:
                latex_exp = latex_exp[:-2] + r'^\dagger$}'
            else:
                latex_exp = latex_exp[:-1] + r'^\dagger$'
        latex_str += latex_exp

    return latex_str


def get_tre_state_of_arts():
    tre = {'Floris': {'Case1': {'TRE_Mean': 1.00, 'TRE_STD': 0.52},
                      'Case2': {'TRE_Mean': 1.02, 'TRE_STD': 0.57},
                      'Case3': {'TRE_Mean': 1.14, 'TRE_STD': 0.89},
                      'Case4': {'TRE_Mean': 1.46, 'TRE_STD': 0.96},
                      'Case5': {'TRE_Mean': 1.61, 'TRE_STD': 1.48},
                      'Case6': {'TRE_Mean': 1.42, 'TRE_STD': 1.71},
                      'Case7': {'TRE_Mean': 1.49, 'TRE_STD': 4.25},
                      'Case8': {'TRE_Mean': 1.62, 'TRE_STD': 1.71},
                      'Case9': {'TRE_Mean': 1.30, 'TRE_STD': 0.76},
                      'Case10': {'TRE_Mean': 1.50, 'TRE_STD': 1.31},
                      'Total': {'TRE_Mean': 1.36, 'TRE_STD': 1.01},
                      },
           'Bob': {'Case1': {'TRE_Mean': 1.27, 'TRE_STD': 1.16},
                   'Case2': {'TRE_Mean': 1.20, 'TRE_STD': 1.12},
                   'Case3': {'TRE_Mean': 1.48, 'TRE_STD': 1.26},
                   'Case4': {'TRE_Mean': 2.09, 'TRE_STD': 1.93},
                   'Case5': {'TRE_Mean': 1.95, 'TRE_STD': 2.10},
                   'Case6': {'TRE_Mean': 5.16, 'TRE_STD': 7.09},
                   'Case7': {'TRE_Mean': 3.05, 'TRE_STD': 3.01},
                   'Case8': {'TRE_Mean': 6.48, 'TRE_STD': 5.37},
                   'Case9': {'TRE_Mean': 2.10, 'TRE_STD': 1.66},
                   'Case10': {'TRE_Mean': 2.09, 'TRE_STD': 2.24},
                   'Total': {'TRE_Mean': 2.64, 'TRE_STD': 4.32},
                   },
           'Koen': {'Case1': {'TRE_Mean': 1.45, 'TRE_STD': 1.06},
                    'Case2': {'TRE_Mean': 1.46, 'TRE_STD': 0.76},
                    'Case3': {'TRE_Mean': 1.57, 'TRE_STD': 1.10},
                    'Case4': {'TRE_Mean': 1.95, 'TRE_STD': 1.32},
                    'Case5': {'TRE_Mean': 2.07, 'TRE_STD': 1.59},
                    'Case6': {'TRE_Mean': 3.04, 'TRE_STD': 2.73},
                    'Case7': {'TRE_Mean': 3.41, 'TRE_STD': 2.75},
                    'Case8': {'TRE_Mean': 2.80, 'TRE_STD': 2.46},
                    'Case9': {'TRE_Mean': 2.18, 'TRE_STD': 1.24},
                    'Case10': {'TRE_Mean': 1.83, 'TRE_STD': 1.36},
                    'Total': {'TRE_Mean': 2.17, 'TRE_STD': 1.89},
                    },
           'Koen_DIR': {'Case1': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case2': {'TRE_Mean': 1.24, 'TRE_STD': 0.61},
                        'Case3': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case4': {'TRE_Mean': 1.70, 'TRE_STD': 1.00},
                        'Case5': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case6': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case7': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case8': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Case9': {'TRE_Mean': 1.61, 'TRE_STD': 0.82},
                        'Case10': {'TRE_Mean': 100, 'TRE_STD': 100},
                        'Total': {'TRE_Mean': 100, 'TRE_STD': 100},
                    },
           'GDL_FIRE': {'Case1': {'TRE_Mean': 1.20, 'TRE_STD': 0.60},
                        'Case2': {'TRE_Mean': 1.19, 'TRE_STD': 0.63},
                        'Case3': {'TRE_Mean': 1.67, 'TRE_STD': 0.90},
                        'Case4': {'TRE_Mean': 2.53, 'TRE_STD': 2.01},
                        'Case5': {'TRE_Mean': 2.06, 'TRE_STD': 1.56},
                        'Case6': {'TRE_Mean': 2.90, 'TRE_STD': 1.70},
                        'Case7': {'TRE_Mean': 3.60, 'TRE_STD': 2.99},
                        'Case8': {'TRE_Mean': 5.29, 'TRE_STD': 5.52},
                        'Case9': {'TRE_Mean': 2.38, 'TRE_STD': 1.46},
                        'Case10': {'TRE_Mean': 2.13, 'TRE_STD': 1.88},
                        'Total': {'TRE_Mean': 2.50, 'TRE_STD': 1.16},
                        },
           }

    return tre


def compare_pair_info_dict(pair_info_ref, pair_info_new, compare_keys=None):
    if compare_keys is None:
        compare_keys = pair_info_ref.keys()
    return all([(pair_info_ref[0][key] == pair_info_new[0][key]) and (pair_info_ref[1][key] == pair_info_new[1][key]) for key in compare_keys])


def merge_landmarks(landmarks_by_IN, IN_test, exp_negative_dvf_list=None):
    """
    This function read landmarks dictionary which for each CN landmarks are there.
    The output is landmarks_merged, in which the IN information is lost and all landmarks are in a list.
    :param landmarks_by_IN: landmarks_merged['exp1'][IN]['DVFAffine']
                                                        ['DVFRegNet']
                                                        ['DVF_nonrigidGroundTruth']
                                                        ['FixedAfterAffineLandmarksWorld']
                                                        [MovingLandmarksWorld]
                                                        ['FixedLandmarksWorld']
    :param IN_test:
    :param exp_negative_dvf_list: in these experiments, the fixed and moving image was swapped, so we need to multiply
                                  the dvf by -1 in order to approximately generate the inverse dvf.
    :return: landmarks_merged:  landmarks_merged['exp1']['DVFAffine']
                                                        ['DVFRegNet']
                                                        ['DVF_nonrigidGroundTruth']
                                                        ['FixedAfterAffineLandmarksWorld']
                                                        [MovingLandmarksWorld]
                                                        ['FixedLandmarksWorld']
    """
    landmarks_merged = {}
    for exp in landmarks_by_IN.keys():
        landmarks_merged[exp] = {}
        for i, IN in enumerate(IN_test):
            for key in landmarks_by_IN[exp][IN].keys():
                if key not in ['setting', 'pair_info', 'network_dict']:
                    if i == 0:
                        landmarks_merged[exp][key] = landmarks_by_IN[exp][IN][key]
                    else:
                        landmarks_merged[exp][key] = np.concatenate((landmarks_merged[exp][key], landmarks_by_IN[exp][IN][key]), axis=0)
        if exp.split('-')[0] in exp_negative_dvf_list:
            landmarks_merged[exp]['DVFRegNet'] = -landmarks_merged[exp]['DVFRegNet']
    return landmarks_merged


def add_affine_exp(landmarks, exp_to_copy=None):
    """
    please note that landmarks is passing by reference. so no need to get the return
    :param landmarks:
    :param exp_to_copy:
    :return:
    """
    if exp_to_copy is None:
        exp_to_copy = next(iter(landmarks))  # first key in the dict
    landmarks['Affine'] = {}
    landmarks['Affine']['TRE'] = [np.linalg.norm(landmarks[exp_to_copy]['MovingLandmarksWorld'][i, :] - landmarks[exp_to_copy]['FixedAfterAffineLandmarksWorld'][i, :])
                                  for i in range(np.shape(landmarks[exp_to_copy]['MovingLandmarksWorld'])[0])]
    landmarks['Affine']['Error'] = [(landmarks[exp_to_copy]['MovingLandmarksWorld'][i, :] - landmarks[exp_to_copy]['FixedAfterAffineLandmarksWorld'][i, :])
                                    for i in range(np.shape(landmarks[exp_to_copy]['MovingLandmarksWorld'])[0])]
    landmarks['Affine']['GroundTruth'] = landmarks[exp_to_copy]['DVF_nonrigidGroundTruth'].copy()
    return landmarks


def add_exp_by_dvf(landmarks, dvf, exp_name, exp_to_copy=None, keys_to_copy=None):
    """
    :param landmarks:
    :param dvf:
    :param exp_name:
    :param exp_to_copy: copy other information from landmarks[exp_to_copy]
    :param keys_to_copy:
    :return:
    """
    if exp_to_copy is None:
        exp_to_copy = next(iter(landmarks))  # first key in the dict
    if keys_to_copy is None:
        keys_to_copy = ['DVFAffine', 'DVF_nonrigidGroundTruth', 'FixedAfterAffineLandmarksWorld', 'MovingLandmarksWorld', 'FixedLandmarksWorld']
    landmarks[exp_name] = {}
    landmarks[exp_name]['DVFRegNet'] = dvf
    for key in keys_to_copy:
        landmarks[exp_name][key] = landmarks[exp_to_copy][key].copy()
    return landmarks


def add_exp_by_error(landmarks, errors, exp_name):
    """
    :param landmarks: landmarks dictionary
    :param errors:  errors value for each landmarks,
    :param exp_name
    :return:
    """
    landmarks[exp_name] = {}
    landmarks[exp_name]['Error'] = np.array(errors)
    landmarks[exp_name]['TRE'] = np.array([np.linalg.norm(errors[i, :]) for i in range(np.shape(errors)[0])])
    return landmarks


def tre_from_dvf(landmarks, exp_list=None):
    """
    :param landmarks:
    :param exp_list:
    :return:
    """
    for exp in exp_list:
        landmarks[exp]['Error'] = np.array([landmarks[exp]['MovingLandmarksWorld'][i, :] -
                                            landmarks[exp]['FixedAfterAffineLandmarksWorld'][i, :] -
                                            landmarks[exp]['DVFRegNet'][i, :]
                                            for i in range(np.shape(landmarks[exp]['MovingLandmarksWorld'])[0])])
        landmarks[exp]['TRE'] = np.array([np.linalg.norm(landmarks[exp]['Error'][i, :])
                                          for i in range(np.shape(landmarks[exp]['MovingLandmarksWorld'])[0])])
    return landmarks


def add_stage_to_experiment_list(exp_list):
    exp_list_stage = []
    for exp_dict in exp_list:
        exp = exp_dict['experiment']
        for stage_str in exp_dict['stage_list']:
            exp = exp + '_' + 'S' + str(stage_str)
        exp_list_stage.append(exp)
    return exp_list_stage


def convert_new_landmark_to_old(landmarks_new, data, cn_list=None, pair_list=None):
    landmarks_old = []
    if data == 'SPREAD':
        landmarks_old = [{} for _ in range(22)]
    elif data in ['DIR-Lab_4D', 'DIR-Lab_COPD']:
        landmarks_old = [{} for _ in range(11)]
    for landmark_dict in landmarks_new:
        pair = [landmark_dict['pair_info'][0]['type_im'], landmark_dict['pair_info'][1]['type_im']]
        cn = landmark_dict['pair_info'][0]['cn']
        if landmark_dict['pair_info'][0]['data'] == data and cn in cn_list and pair in pair_list:
            for key in landmark_dict.keys():
                landmarks_old[cn][key] = copy.deepcopy(landmark_dict[key])

    return landmarks_old


def fancy_names(landmarks, exp_list=None):
    for exp in exp_list:
        landmarks[exp]['FancyName'] = su.fancy_exp_name(exp)
    return landmarks
