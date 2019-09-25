import numpy as np
import os
import pickle
import SimpleITK as sitk
import functions.reading.real_pair as real_pair
import functions.setting.setting_utils as su


def landmark_info(setting, pair_info, base_reg='Affine'):
    """
    extract landmark information. Be very careful about the order:
    :param setting:
    :param pair_info:
    :return:
    'FixedLandmarksWorld': xyz order
    'MovingLandmarksWorld': xyz order
    'FixedAfterAffineLandmarksWorld': xyz order
    'FixedLandmarksIndex': xyz order
    """
    pair = real_pair.Images(setting, pair_info, stage=1)
    pair.prepare_for_landmarks(padding=False)
    current_landmark = {'setting': setting,
                        'pair_info': pair_info,
                        'FixedLandmarksWorld': pair._fixed_landmarks_world.copy(),
                        'MovingLandmarksWorld': pair._moving_landmarks_world.copy(),
                        'FixedAfter'+base_reg+'LandmarksWorld': pair._fixed_after_affine_landmarks_world.copy(),
                        'DVF'+base_reg: pair._dvf_affine.copy(),
                        'DVF_nonrigidGroundTruth': pair._moving_landmarks_world - pair._fixed_after_affine_landmarks_world,
                        'FixedLandmarksIndex': pair._fixed_landmarks_index.copy(),
                        }

    return current_landmark


def landmarks_from_dvf_old(setting, IN_test_list):
    # %%------------------------------------------- Setting of generating synthetic DVFs------------------------------------------
    saved_file = su.address_generator(setting, 'landmarks_file')
    if os.path.isfile(saved_file):
        raise ValueError('cannot overwrite, please change the name of the pickle file: ' + saved_file)

    # %%------------------------------------------------------  running   ---------------------------------------------------------
    landmarks = [{} for _ in range(22)]
    for IN in IN_test_list:
        image_pair = real_pair.Images(IN, setting=setting)
        image_pair.prepare_for_landmarks(padding=False)
        landmarks[IN]['FixedLandmarksWorld'] = image_pair._fixed_landmarks_world.copy()
        landmarks[IN]['MovingLandmarksWorld'] = image_pair._moving_landmarks_world.copy()
        landmarks[IN]['FixedAfterAffineLandmarksWorld'] = image_pair._fixed_after_affine_landmarks_world.copy()
        landmarks[IN]['DVFAffine'] = image_pair._dvf_affine.copy()
        landmarks[IN]['DVF_nonrigidGroundTruth'] = image_pair._moving_landmarks_world - image_pair._fixed_after_affine_landmarks_world
        landmarks[IN]['FixedLandmarksIndex'] = image_pair._fixed_landmarks_index
        dvf_s0_sitk = sitk.ReadImage(su.address_generator(setting, 'dvf_s0', cn=IN))
        dvf_s0 = sitk.GetArrayFromImage(dvf_s0_sitk)
        landmarks[IN]['DVFRegNet'] = np.stack([dvf_s0[image_pair._fixed_landmarks_index[i, 2],
                                                      image_pair._fixed_landmarks_index[i, 1],
                                                      image_pair._fixed_landmarks_index[i, 0]]
                                               for i in range(len(image_pair._fixed_landmarks_index))])
        print('CPU: IN = {} is done in '.format(IN))
    with open(saved_file, 'wb') as f:
        pickle.dump(landmarks, f)

