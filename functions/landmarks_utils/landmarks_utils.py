import copy
import logging
import numpy as np
import os
import pickle
import time
import functions.setting.setting_utils as su
from .image import landmarks_from_dvf


def calculate_landmark(setting, pair_info, network_dict, overwrite_landmarks=False):
    time_before = time.time()
    stage_list = setting['ImagePyramidSchedule']
    landmark_address = su.address_generator(setting, 'landmarks_file', stage_list=stage_list)
    if os.path.isfile(landmark_address):
        with open(landmark_address, 'rb') as f:
            landmark = pickle.load(f)
    else:
        landmark = []

    if any([(dict(sorted(pair_info[0].items())) == dict(sorted(landmark_i['pair_info'][0].items()))) and
            (dict(sorted(pair_info[1].items())) == dict(sorted(landmark_i['pair_info'][1].items())))
            for landmark_i in landmark]):
        # the above is just a simple comparison of two dict. It should be noted that if the keys are not sorted,
        # then == results in False value.
        if not overwrite_landmarks:
            logging.debug('Landmark skipping stage_list={}'.format(stage_list) + ' Fixed: ' + pair_info[0]['data'] +
                          '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '  Moving:' +
                          pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(
                pair_info[1]['cn'], pair_info[1]['type_im']))
            return 1
        else:
            logging.debug('Landmark overwriting stage_list={}'.format(stage_list) + ' Fixed: ' + pair_info[0]['data'] +
                          '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im']) + '  Moving:' +
                          pair_info[1]['data'] + '_CN{}_TypeIm{}'.format(
                pair_info[1]['cn'], pair_info[1]['type_im']))

    landmark_dict = landmarks_from_dvf(setting, pair_info)
    landmark_dict['network_dict'] = copy.deepcopy(network_dict)
    landmark.append(landmark_dict)

    with open(landmark_address, 'wb') as f:
        pickle.dump(landmark, f)
    time_after = time.time()
    logging.debug('Landmark stage_list={}'.format(stage_list)+' Fixed: '+pair_info[0]['data']+
                  '_CN{}_TypeIm{},'.format(pair_info[0]['cn'], pair_info[0]['type_im'])+'  Moving:' +
                  pair_info[1]['data']+'_CN{}_TypeIm{} is done in {:.2f}s '.format(
        pair_info[1]['cn'], pair_info[1]['type_im'], time_after - time_before))
    return landmark


def merge_landmarks(landmarks_by_IN, IN_test, exp_negative_dvf_list=[]):
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
    elif data == 'DIR-Lab_4D':
        landmarks_old = [{} for _ in range(10)]
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
