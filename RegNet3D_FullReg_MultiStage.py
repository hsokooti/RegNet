import argparse
import copy
import json
import logging
import os
import shutil
import functions.elastix_python as elx
import functions.setting_utils as su
import functions.landmarks_utils as lu
import functions.general_utils as gut
import functions.registration as reg


def full_registration_multi_stage(stage_list=None, current_experiment=None):
    """
    Perform registration in a multi-stage fashion.
    key features:
        for each stage you can have different saved model
        for each stage you can have different patch size
    :param stage_list: Total number of stages. It can be selected between: [4, 2, 1]: three stages, [2, 1]: two stages, [1]: one stages
    """

    # %%----------------------------------------------- General parameters -----------------------------------------------
    if current_experiment is None:
        current_experiment = '2020_multistage_crop3'  # 2020_multistage_crop3, 2020_multistage_dec3
    if stage_list is None:
        stage_list = [4, 2, 1]
    landmark_calculation = True
    overwrite_dvf = False
    overwrite_landmarks = False

    # %%------------------------------------------------ Data description ------------------------------------------------
    # {'data': 'SPREAD',
    #  'TestingCNList': [i for i in range(13, 22) if i not in [14, 19]],
    #  'TestingTypeImList': [0, 1],
    #  'TestingPairList': [[0, 1]]
    #  },
    data_exp_dict = [
                     {'data': 'DIR-Lab_4D',
                      'TestingCNList': [4, 5, 6, 7, 8, 9, 10],
                      'TestingPairList': [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]],
                      'TrainingCNList': [1, 2, 3],
                      'TrainingPairList': [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
                      }
                     ]
    train_mode = 'Training'
    setting, backup_folder = initialize(current_experiment, stage_list)
    setting = su.load_setting_from_data_dict(setting, data_exp_dict)

    # %%----------------------------------- Network description for each stage------------------------------------------------
    if current_experiment == '2020_multistage_dec3':
        network_dict = {'Stage1': {'NetworkDesign': 'decimation3',
                                   'NetworkLoad': '20181030_225202_sh',
                                   'R': 127,
                                   'Ry': 63,
                                   'Ry_erode': 2,
                                   'Stage': 1
                                   },
                        'Stage2': {'NetworkDesign': 'decimation3',
                                   'NetworkLoad': '20181030_225310_sh',
                                   'R': 127,
                                   'Ry': 63,
                                   'Ry_erode': 2,
                                   'Stage': 2
                                   },
                        'Stage4': {'NetworkDesign': 'decimation3',
                                   'NetworkLoad': '20181017_max20_S4_dec3',
                                   'R': 127,
                                   'Ry': 63,
                                   'Ry_erode': 2,
                                   'Stage': 4
                                   },
                        }
    if current_experiment == '2020_multistage_crop3':
        network_dict = {'Stage1': {'NetworkDesign': 'crop3_connection',
                                   'NetworkLoad': '20181008_max7_D9_crop3_A',
                                   'R': 100,
                                   'Ry': 60,
                                   'Ry_erode': 2,
                                   'Stage': 1
                                   },
                        'Stage2': {'NetworkDesign': 'crop3_connection',
                                   'NetworkLoad': '20180921_max15_D12_stage2_crop3',
                                   'R': 100,
                                   'Ry': 60,
                                   'Ry_erode': 2,
                                   'Stage': 2
                                   },
                        'Stage4': {'NetworkDesign': 'crop3_connection',
                                   'NetworkLoad': '20181009_max20_D12_stage4_crop3',
                                   'R': 100,
                                   'Ry': 60,
                                   'Ry_erode': 2,
                                   'Stage': 4
                                   },
                        }

    # %%------------------------------------------------ General Setting ------------------------------------------------
    setting['normalization'] = ''           # 'linear' The method to normalize the intensities
    setting['read_pair_mode'] = 'real'      # 'real', 'synthetic'
    setting['ImagePyramidSchedule'] = stage_list
    setting['PaddingForDownSampling'] = 'constant'  # 'constant': Setting['defaultPixelValue'], 'mirror': not implemented
    setting['WriteAfterEachStage'] = True  # Detailed writing of images: DVF and Deformed images after
    setting['verbose'] = True               # Detailed printing
    setting['verbose_image'] = True  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['torsoMask'] = True

    # %%---------------------------------------------- Registration Setting -----------------------------------------------
    setting['reg_AffineParameter'] = 'Par0011.affine.txt'
    setting['reg_BSplineParameter'] = 'Par0011.bspline.txt'
    setting['reg_UseMask'] = False
    setting['reg_NumberOfThreads'] = 7

    # Load global_step for each network
    for stage in setting['ImagePyramidSchedule']:
        network_dict['Stage'+str(stage)]['GlobalStepLoad'] = su.load_global_step_from_current_experiment(network_dict['Stage'+str(stage)]['NetworkLoad'])
        if not network_dict['Stage'+str(stage)]['GlobalStepLoad']:
            raise ValueError('GlobalStepLoad is not found for the network:' + network_dict['Stage'+str(stage)]['NetworkLoad'])

    # Serious warning about overwrite
    if overwrite_dvf or overwrite_landmarks:
        'comment'
        # if not gut.io.query_yes_no('overwrite is set to TRUE and old results might be overwritten. Continue?'):
        #     raise RuntimeError('Interrupted by user. You might change the current_experiment to prevent overwriting problem ')

    # check stage_list
    if stage_list not in [[4, 2, 1], [2, 1], [1]]:
        raise ValueError('In the current implementation stage_list can be only be selected between: '
                         '[4, 2, 1]: three stages, [2, 1]: two stages and [1]: one stages ')

    su.write_setting(setting, setting_address=backup_folder+'setting.txt')
    with open(backup_folder+'network.txt', 'w') as file:
        file.write(json.dumps(network_dict, sort_keys=True, indent=4, separators=(',', ': ')))

    # %%------------------------------------------- Running multi-stage------------------------------------------
    pair_info_list = su.get_im_info_list_from_train_mode(setting, train_mode=train_mode, load_mode='Pair')

    for pair_info in pair_info_list:
        # elx.affine(setting, pair_info, setting['reg_AffineParameter'], stage=1, overwrite=False)
        # elx.affine_transformix_points(setting, pair_info, stage=1, overwrite=False)
        # elx.affine_transformix_torso(setting, pair_info, stage=1, overwrite=False)
        reg.multi_stage(setting, network_dict=network_dict, pair_info=pair_info, overwrite=overwrite_dvf)
        if landmark_calculation:
            lu.calculate_landmark(setting, pair_info=pair_info, network_dict=network_dict,
                                  overwrite_landmarks=overwrite_landmarks)


def initialize(current_experiment, stage_list):
    parser = argparse.ArgumentParser(description='read where_to_run')
    parser.add_argument('--where_to_run', '-w', help='This is an optional argument, you choose between "Auto" or "Cluster". The default value is "Auto"')
    args = parser.parse_args()
    where_to_run = args.where_to_run
    setting = su.initialize_setting(current_experiment=current_experiment, where_to_run=where_to_run)
    backup_number = 1
    backup_root_folder = su.address_generator(setting, 'result_step_folder', stage_list=stage_list)
    backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    while os.path.isdir(backup_folder):
        backup_number = backup_number + 1
        backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    gut.logger.set_log_file(backup_folder+'log.txt', short_mode=True)
    shutil.copy(os.path.realpath(__file__), backup_folder+os.path.realpath(__file__).rsplit('/', maxsplit=1)[1])
    return setting, backup_folder


if __name__ == '__main__':
    stage_list_loop = [[4, 2, 1], [2, 1], [1]]
    stage_list_loop = [[1]]
    current_experiment_loop_archive = ['2020_multistage_crop3', '2020_multistage_dec3']
    current_experiment_loop = current_experiment_loop_archive
    for current_experiment_exp in current_experiment_loop:
        for stage_list_exp in stage_list_loop:
            full_registration_multi_stage(stage_list=stage_list_exp, current_experiment=current_experiment_exp)
