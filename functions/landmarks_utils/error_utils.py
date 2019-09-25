import copy
import dill
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import wilcoxon
import SimpleITK as sitk
import time
import xlsxwriter
import functions.setting.setting_utils as su
from .landmarks_utils import compare_pair_info_dict
from .image import landmark_info
import functions.registration as reg


def calculate_write_landmark(setting, pair_info, overwrite_landmarks=False, overwrite_landmarks_hard=False, base_reg=None):
    """
    Add the following information:     [setting, 'pair_info', 'FixedLandmarksWorld', 'MovingLandmarksWorld', 'FixedAfterAffineLandmarksWorld'
                                        'DVFAffine', 'DVF_nonrigidGroundTruth', 'FixedLandmarksIndex']


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
    landmark_address = su.address_generator(setting, 'landmarks_file', stage_list=stage_list, current_experiment=setting['lstm_exp'], step=setting['network_lstm_dict']['GlobalStepLoad'])
    if os.path.isfile(landmark_address):
        with open(landmark_address, 'rb') as f:
            landmark = dill.load(f)
    else:
        landmark = []
        result_landmarks_folder = su.address_generator(setting, 'result_landmarks_folder', stage_list=stage_list, current_experiment=setting['lstm_exp'], step=setting['network_lstm_dict']['GlobalStepLoad'])
        if not os.path.isdir(result_landmarks_folder):
            os.makedirs(result_landmarks_folder)

    calculate_multi_stage_error = False

    pair_info_text = 'stage_list={}'.format(stage_list) + ' Fixed: ' + pair_info[0]['data'] + \
                     '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '  Moving:' + \
                     pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(pair_info[1]['cn'], pair_info[1]['type_im'])
    ind_find_list = [compare_pair_info_dict(pair_info, landmark_i['pair_info'], compare_keys=['data', 'cn', 'type_im'])
                     for landmark_i in landmark]
    ind_find = None
    if any(ind_find_list):
        ind_find = ind_find_list.index(True)
        if not overwrite_landmarks:
            logging.info('Skipping ' + pair_info_text)
        else:
            calculate_multi_stage_error = True
            logging.info('overwriting ' + pair_info_text)

    if ind_find is None:
        ind_find = -1
        calculate_multi_stage_error = True

    # if overwrite_landmarks_hard:
    #     landmark_dict = landmark_info(setting, pair_info)
    #     keys_to_copy = ['FixedLandmarksWorld', 'MovingLandmarksWorld', 'FixedAfterAffineLandmarksWorld',
    #                     'DVFAffine', 'DVF_nonrigidGroundTruth', 'FixedLandmarksIndex']
    #     for key in keys_to_copy:
    #         landmark[ind_find][key] = copy.deepcopy(landmark_dict[key])

    if calculate_multi_stage_error or overwrite_landmarks_hard:
        landmark_dict = landmark_info(setting, pair_info, base_reg=base_reg)
        landmark_dict = reg.multi_stage_error(setting, landmark_dict, pair_info=pair_info)
        # landmark_dict = landmark_info(setting, pair_info)
        landmark_dict['network_dict'] = copy.deepcopy(setting['network_dict'])
        landmark_dict['network_lstm_dict'] = copy.deepcopy(setting['network_lstm_dict'])
        landmark.append(landmark_dict)

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
        exp_pure = exp_dict['experiment'].rsplit('/')[1]
        stage_list = exp_dict['stage_list']
        exp = exp_dict['experiment']

        exp_folder = exp.split('-')[0]
        exp_key_name = exp + '_' +exp_dict['BaseReg']
        landmark_address = su.address_generator(setting, 'landmarks_file', current_experiment=exp_folder, stage_list=stage_list, base_reg=exp_dict['BaseReg'], step=exp_dict['GlobalStepLoad'])
        with open(landmark_address, 'rb') as f:
            landmarks_load = dill.load(f)
            # landmarks_load = dill.load(f)
        landmarks_dict[exp_key_name] = []
        for pair_info in pair_info_list:
            landmark_pair = dict()
            pair_info_text = exp_key_name + ' Fixed: ' + pair_info[0]['data'] + \
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

            landmark_all_info = {'pair_info': copy.deepcopy(pair_info),
                                 'landmark_info': landmark_pair}

            landmarks_dict[exp_key_name].append(landmark_all_info)
    return landmarks_dict


def table_box_plot(setting, landmarks, exp_list, fig_measure_list=None, plot_per_pair=False, fig_ext='.png',
                   plot_folder=None, paper_table=None, naming_strategy=None, jacobian=False, label_times2=None,
                   label_times1=None, step=0, xlx_name=None):
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

    landmarks_merged = dict()
    if plot_folder is None:
        plot_key = list(landmarks.items())[-1][0]
    else:
        plot_key = plot_folder
    if xlx_name is None:
        xlx_name = 'results'
    stage_list = [4, 2, 1]


    result_folder = su.address_generator(setting, 'result_detail_folder',
                                         current_experiment=plot_key,
                                         stage_list=stage_list,
                                         step=step,
                                         pair_info=landmarks[plot_key + '_' + exp_list[0]['BaseReg']][0]['pair_info']
                                         )
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    xlsx_address = result_folder + xlx_name+'.xlsx'
    # if os.path.isfile(xlsx_address):
    #     raise ValueError(xlsx_address + 'already exists cannot overwrite')
    workbook = xlsxwriter.Workbook(xlsx_address)
    worksheet = workbook.add_worksheet()
    line = 0
    header = {'exp': 0, 'P0A0': 1, 'P0A1': 2, 'P0A2': 3, 'P1A0': 4, 'P1A1': 5,
              'P1A2': 6, 'P2A0': 7, 'P2A1': 8, 'P2A2': 9, 'A0': 10, 'A1': 11,
              'A2': 12, 'F0': 13, 'F1': 14, 'F2': 15, 'Acc': 16
            }

    for key in header.keys():
        worksheet.write(line, header[key], key)
    num_exp = len(landmarks.keys())
    for exp_i, exp in enumerate(landmarks.keys()):
        landmarks_merged[exp] = {'DVF_error_times2_label': np.empty([0]),
                                 'DVF_error_times2_logits': np.empty([0,  setting['NumberOfLabels']]),
                                 'DVF_error_times1_label': np.empty([0]),
                                 'DVF_error_times1_logits': np.empty([0,  setting['NumberOfLabels']]),
                                 'DVF_error_times0_label': np.empty([0]),
                                 'DVF_error_times0_logits': np.empty([0,  setting['NumberOfLabels']]),
                                 'DVF_nonrigidGroundTruth_magnitude': np.empty([0]),
                                 'CleanName': su.clean_exp_name(exp),
                                 'FancyName': su.fancy_exp_name(exp),
                                 }

        num_pair = len(landmarks[exp])
        for pair_i, landmark_pair in enumerate(landmarks[exp]):
            pair_info = landmark_pair['pair_info']
            pair_info_text = landmarks_merged[exp]['CleanName'] + '_Fixed_' + pair_info[0]['data'] + \
                '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '_Moving_' + \
                pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(pair_info[1]['cn'], pair_info[1]['type_im'])
            for i in range(3):
                if 'DVF_error_times'+str(i)+'_logits' in landmark_pair['landmark_info'].keys():
                    landmarks_merged[exp]['DVF_error_times'+str(i)+'_logits'] = np.vstack((landmarks_merged[exp]['DVF_error_times'+str(i)+'_logits'],
                                                                                           landmark_pair['landmark_info']['DVF_error_times'+str(i)+'_logits']))
                    landmarks_merged[exp]['DVF_error_times'+str(i)+'_label'] = np.append(landmarks_merged[exp]['DVF_error_times'+str(i)+'_label'],
                                                                                         landmark_pair['landmark_info']['DVF_error_times'+str(i)+'_label'])
            landmarks_merged[exp]['DVF_nonrigidGroundTruth_magnitude'] = np.append(landmarks_merged[exp]['DVF_nonrigidGroundTruth_magnitude'],
                                                                                   landmark_pair['landmark_info']['DVF_nonrigidGroundTruth_magnitude'])

            measure = calculate_measure(landmark_pair['landmark_info'], label_times2=label_times2, label_times1=label_times1)
            measure['exp'] = pair_info_text
            if plot_per_pair:
                print_latex(measure)
            line = exp_i + pair_i * (num_exp + 1) + 1
            for key in header.keys():
                if key in measure.keys():
                    worksheet.write(line, header[key], measure[key])
                    landmark_pair['landmark_info'][key] = measure[key]

        measure_merged = calculate_measure(landmarks_merged[exp], label_times2=label_times2, label_times1=label_times1)
        if naming_strategy == 'Clean':
            measure_merged['exp'] = su.clean_exp_name(exp)
        elif naming_strategy == 'Fancy':
            measure_merged['exp'] = su.fancy_exp_name(exp)
        else:
            measure_merged['exp'] = exp
        # print_latex(measure_merged)
        line = exp_i + num_pair * (num_exp + 1) + 2
        for key in header.keys():
            if key in measure.keys():
                if key in header.keys() and key in measure_merged.keys():
                    worksheet.write(line, header[key], measure_merged[key])

    full_merge = {'DVF_error_times2_label': np.empty([0]),
                  'DVF_nonrigidGroundTruth_magnitude': np.empty([0]),
                  'DVF_error_times2_logits': np.empty([0,  setting['NumberOfLabels']]),
                  }
    for exp_i, exp in enumerate(landmarks_merged.keys()):
        full_merge['DVF_error_times2_logits'] = np.vstack((full_merge['DVF_error_times2_logits'], landmarks_merged[exp]['DVF_error_times2_logits']))
        full_merge['DVF_error_times2_label'] = np.append(full_merge['DVF_error_times2_label'], landmarks_merged[exp]['DVF_error_times2_label'])
        full_merge['DVF_nonrigidGroundTruth_magnitude'] = np.append(full_merge['DVF_nonrigidGroundTruth_magnitude'], landmarks_merged[exp]['DVF_nonrigidGroundTruth_magnitude'])
    measure_full_merged = calculate_measure(full_merge, label_times2=label_times2, label_times1=label_times1)
    measure_full_merged['exp'] = 'Total'
    line = line+1
    for key in header.keys():
        if key in measure_full_merged.keys():
            worksheet.write(line, header[key], measure_full_merged[key])
    workbook.close()

    # for measure in fig_measure_list:
    #     if plot_per_pair:
    #         for pair_i in range(len(landmarks[next(iter(landmarks))])):
    #             fig, ax = plt.subplots(figsize=(15, 8))
    #             bplot1 = plt.boxplot([landmarks[exp][pair_i]['landmark_info'][measure] for exp in landmarks.keys()],
    #                                  patch_artist=True, notch=True)
    #             title_name = landmarks[next(iter(landmarks))][pair_i]['landmark_info']['exp']
    #             plt.title(title_name)
    #             plt.draw()
    #             plt.savefig(result_folder+measure+'_'+title_name+fig_ext)
    #             plt.close()
    #
    #     fig, ax = plt.subplots(figsize=(15, 8))
    #     bplot1 = plt.boxplot([landmarks_merged[exp][measure] for exp in landmarks_merged.keys()],
    #                          patch_artist=True, notch=True)
    #     title_name = measure + '_Merged'
    #     plt.title(title_name)
    #     plt.savefig(result_folder + measure + '_' + title_name + fig_ext)
    #     plt.draw()
    #     plt.close()


def calculate_measure(input_dict, label_times2=None, label_times1=None):
    measure = dict()
    dvf_ground_truth_label = copy.deepcopy(input_dict['DVF_nonrigidGroundTruth_magnitude'])
    dvf_ground_truth_label[dvf_ground_truth_label <= 3] = 0
    dvf_ground_truth_label[(3 < dvf_ground_truth_label) & (dvf_ground_truth_label <= 6)] = 1
    dvf_ground_truth_label[6 < dvf_ground_truth_label] = 2
    dvf_ground_truth_label = dvf_ground_truth_label.astype(np.int)

    dvf_error_times2_label = copy.deepcopy(input_dict['DVF_error_times2_label'].astype(np.int))
    if label_times2 == [0, 1, [2, 3]]:
        dvf_error_times2_label[dvf_error_times2_label == 3] = 2
    elif label_times2 == [[0, 1], 2, 3]:
        dvf_error_times2_label[dvf_error_times2_label == 1] = 0
        dvf_error_times2_label[dvf_error_times2_label == 2] = 1
        dvf_error_times2_label[dvf_error_times2_label == 3] = 2
    else:
        raise ValueError('label_times2 not defined')

    for p in range(3):
        predicted_indices = np.where(dvf_error_times2_label == p)
        for a in range(3):
            actual_indices = np.where(dvf_ground_truth_label[predicted_indices[0]] == a)
            measure['P'+str(p)+'A'+str(a)] = len(actual_indices[0])

    for i in range(3):
        measure['P'+str(i)] = measure['P'+str(i)+'A0'] + measure['P'+str(i)+'A1'] + measure['P'+str(i)+'A2']
        measure['A'+str(i)] = measure['P0'+'A'+str(i)] + measure['P1'+'A'+str(i)] + measure['P2'+'A'+str(i)]
        if measure['P'+str(i)] != 0:
            measure['Pr'+str(i)] = measure['P'+str(i)+'A'+str(i)] / measure['P'+str(i)] * 100
        else:
            measure['Pr'+str(i)] = 0
        if measure['A'+str(i)] != 0:
            measure['R'+str(i)] = measure['P'+str(i)+'A'+str(i)] / measure['A'+str(i)] * 100
        else:
            measure['R'+str(i)] = 0
        if measure['Pr'+str(i)] * measure['R'+str(i)] != 0:
            measure['F'+str(i)] = 2 * measure['Pr'+str(i)] * measure['R'+str(i)] / (measure['Pr'+str(i)] + measure['R'+str(i)])
        else:
            measure['F' + str(i)] = 0
    measure['Acc'] = (measure['P0A0'] + measure['P1A1'] + measure['P2A2']) / (measure['A0'] + measure['A1'] + measure['A2']) * 100
    return measure
