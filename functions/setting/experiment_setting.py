def load_global_step_from_predefined_list(exp):
    global_step_dict = {'20180124_max20_2M': '1155015',
                        }

    if exp not in global_step_dict.keys():
        global_step_dict[exp] = '0'
    global_step = global_step_dict[exp]
    return global_step


def fancy_exp_name(exp):
    fancy_dict = {'Affine': 'Affine',
                  'Bspline_1S': 'Bspline 1S',
                  'Bspline_3S': 'Bspline 3S',
                  }

    if exp not in fancy_dict.keys():
        fancy_dict[exp] = exp
    fancy_name = fancy_dict[exp]
    return fancy_name


def clean_exp_name(exp):
    fancy_dict = {'Affine_S4_S2_S1': 'Affine',
                  'Bspline_1S': 'Bspline 1S',
                  'Bspline_3S': 'Bspline 3S',
                  }
    if exp not in fancy_dict.keys():
        fancy_dict[exp] = exp
    fancy_name = fancy_dict[exp]
    return fancy_name


def load_network_multi_stage_from_predefined(current_experiment):
    network_dict = None

    if current_experiment in ['2020_multistage_crop4_K_NoResp_more_itr']:
        network_dict = {'stage1': {'NetworkDesign': 'crop4_connection',
                                   'NetworkLoad': '20190211_3D_max7_D14_K_NoResp_S1_crop4',
                                   'R': 'Auto',
                                   'Ry': 'Auto',
                                   'Ry_erode': 'Auto',
                                   'GlobalStepLoad': 'Last',
                                   'MaskToZero': 'Lung',
                                   'stage': 1
                                   },
                        'stage2': {'NetworkDesign': 'crop4_connection',
                                   'NetworkLoad': '20190211_3D_max15_D14_K_NoResp_S2_r3_crop4',
                                   'R': 'Auto',
                                   'Ry': 'Auto',
                                   'Ry_erode': 'Auto',
                                   'GlobalStepLoad': 'Last',
                                   'MaskToZero': 'Lung',
                                   'stage': 2
                                   },
                        'stage4': {'NetworkDesign': 'crop4_connection',
                                   'NetworkLoad': '20190227_3D_max20_D14_K_NoResp_S4_crop4',
                                   'R': 'Auto',
                                   'Ry': 'Auto',
                                   'Ry_erode': 'Auto',
                                   'GlobalStepLoad': 'Last',
                                   'MaskToZero': 'Lung',
                                   'stage': 4
                                   },
                        }
    return network_dict
