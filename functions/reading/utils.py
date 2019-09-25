import copy
import functions.setting.setting_utils as su
from joblib import Parallel, delayed
import json
import logging
import multiprocessing
import numpy as np
import os
import time


def search_indices(dvf, c, class_balanced, margin, dim_im, torso):
    """
    This function searches for voxels based on the ClassBalanced in the parallel mode: if Setting['ParallelSearching'] == True

    :param dvf:           input DVF
    :param c:                   enumerate of the class (in for loop over all classes)
    :param class_balanced:      a vector indicates the classes, for instance [a,b] implies classes [0,a), [a,b)
    :param margin:              Margin of the image. so no voxel would be selected if the index is smaller than K or greater than (ImageSize - K)
    :param dim_im:             '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction. Hence, we can't use np.all and we have to use np.any.
    :param torso:

    :return:                I1 which is a numpy array of ravel_multi_index

    Hessam Sokooti h.sokooti@gmail.com
    """

    mask = np.zeros(np.shape(dvf)[:-1], dtype=np.bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    if torso is not None:
        mask = mask & torso
    i1 = None
    if c == 0:
        # Future: you can add a mask here to prevent selecting pixels twice!
        i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf) < class_balanced[c]), axis=3)) & mask), np.shape(dvf)[:-1]).astype(np.int32)
        # the output of np.where occupy huge part of memory! by converting it to a numpy array lots of memory can be saved!
    elif (c > 0) & (c < len(class_balanced)):
        if dim_im == 2:
            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf) < class_balanced[c]), axis=3)) & (np.any((np.abs(dvf) >= class_balanced[c - 1]), axis=3)) &
                                               mask), np.shape(dvf)[:-1]).astype(np.int32)
        elif dim_im == 3:
            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf) < class_balanced[c]), axis=3)) & (np.all((np.abs(dvf) >= class_balanced[c - 1]), axis=3)) &
                                               mask), np.shape(dvf)[:-1]).astype(np.int32)
    return i1


def search_indices_seq(dvf_label, c, class_balanced, margin, torso):
    """
    This function searches for voxels based on the ClassBalanced in the parallel mode: if Setting['ParallelSearching'] == True

    :param dvf_label:           input DVF
    :param c:                   enumerate of the class (in for loop over all classes)
    :param class_balanced:      a vector indicates the classes, for instance [a,b] implies classes [0,a), [a,b)
    :param margin:              Margin of the image. so no voxel would be selected if the index is smaller than K or greater than (ImageSize - K)
    :param dim_im:             '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction. Hence, we can't use np.all and we have to use np.any.

    :return:                I1 which is a numpy array of ravel_multi_index

    Hessam Sokooti h.sokooti@gmail.com
    """

    mask = np.zeros(np.shape(dvf_label), dtype=np.bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    if torso is not None:
        mask = mask & torso
    if isinstance(class_balanced[c], list):
        if len(class_balanced[c]) == 2:
            i1 = np.ravel_multi_index(np.where(np.logical_and(np.logical_or(
                dvf_label == class_balanced[c][0], dvf_label == class_balanced[c][1]), mask)), np.shape(dvf_label)).astype(np.int32)
        else:
            raise ValueError('implemented for maximum of two values per class')
    else:
        i1 = np.ravel_multi_index(np.where(np.logical_and(dvf_label == class_balanced[c], mask)), np.shape(dvf_label)).astype(np.int32)
    # the output of np.where occupy huge part of memory! by converting it to a numpy array lots of memory can be saved!
    return i1


def shuffled_indices_from_chunk(setting, dvf_list=None, torso_list=None, im_info_list=None, stage=None, stage_sequence=None,
                                semi_epoch=None, chunk=None, samples_per_image=None, log_header='', full_image=None, seq_mode=False,
                                chunk_length_force_to_multiple_of=None):
    if full_image:
        ishuffled = np.arange(len(dvf_list))
    else:
        if seq_mode:
            ishuffled = shuffled_indices_from_chunk_patch_seq(setting, dvf_list=dvf_list, torso_list=torso_list,
                                                              stage_sequence=stage_sequence, semi_epoch=semi_epoch, chunk=chunk,
                                                              samples_per_image=samples_per_image, log_header=log_header,
                                                              chunk_length_force_to_multiple_of=chunk_length_force_to_multiple_of)
        else:
            ishuffled = shuffled_indices_from_chunk_patch(setting, dvf_list=dvf_list, torso_list=torso_list, im_info_list=im_info_list,
                                                          stage=stage, semi_epoch=semi_epoch, chunk=chunk, samples_per_image=samples_per_image,
                                                          log_header=log_header)
    return ishuffled


def shuffled_indices_from_chunk_patch(setting, dvf_list=None, torso_list=None, im_info_list=None, stage=None, semi_epoch=None,
                                      chunk=None, samples_per_image=None, log_header=''):
    for single_dict in setting['DataExpDict']:
        iclass_folder = su.address_generator(setting, 'IClassFolder', data=single_dict['data'], deform_exp=single_dict['deform_exp'], stage=stage)
        if not (os.path.isdir(iclass_folder)):
            os.makedirs(iclass_folder)

    margin = setting['Margin']
    class_balanced = setting['ClassBalanced']
    indices = {}
    for c in range(len(class_balanced)):
        indices['class'+str(c)] = []
    start_time = time.time()
    if setting['ParallelSearching']:
        num_cores = multiprocessing.cpu_count() - 2
        results = [None] * len(dvf_list) * len(class_balanced)
        count_iclass_loaded = 0
        for i_dvf, im_info in enumerate(im_info_list):
            for c in range(len(class_balanced)):
                iclass_address = su.address_generator(setting, 'IClass', data=im_info['data'], deform_exp=im_info['deform_exp'], cn=im_info['cn'],
                                                      type_im=im_info['type_im'], dsmooth=im_info['dsmooth'], c=c, stage=stage)
                if os.path.isfile(iclass_address):
                    results[i_dvf * len(class_balanced) + c] = np.load(iclass_address)  # double checked
                    count_iclass_loaded += 1
        if count_iclass_loaded != len(results):
            logging.debug(log_header+': not all I1 found. start calculating... SemiEpoch = {}, Chunk = {}, stage={}'.format(semi_epoch, chunk, stage))
            results = Parallel(n_jobs=num_cores)(delayed(search_indices)(dvf=dvf_list[i], torso=torso_list[i],
                                                                         c=c, class_balanced=class_balanced, margin=margin,
                                                                         dim_im=setting['Dim'])
                                                 for i in range(0, len(dvf_list)) for c in range(0, len(class_balanced)))
            for i_dvf, im_info in enumerate(im_info_list):
                for c in range(0, len(class_balanced)):
                    iclass_address = su.address_generator(setting, 'IClass', data=im_info['data'], deform_exp=im_info['deform_exp'], cn=im_info['cn'],
                                                          type_im=im_info['type_im'], dsmooth=im_info['dsmooth'], c=c, stage=stage)
                    np.save(iclass_address, results[i_dvf * len(class_balanced) + c])  # double checked
        for iresults in range(0, len(results)):
            i_dvf = iresults // (len(class_balanced))  # first loop in the Parallel: for i in range(0, len(dvf_list))
            c = iresults % (len(class_balanced))   # second loop in the Parallel: for j in range(0, len(class_balanced)+1)
            if len(results[iresults]):
                if len(indices['class'+str(c)]) == 0:
                    indices['class'+str(c)] = np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])
                else:
                    indices['class'+str(c)] = np.concatenate((indices['class'+str(c)], np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])), axis=0)
        del results
        end_time = time.time()
        if setting['verbose']:
            logging.debug(log_header+' Parallel searching for {} classes is Done in {:.2f}s'.format(len(class_balanced), end_time - start_time))
    else:
        for i_dvf, im_info in enumerate(im_info_list):
            mask = np.zeros(np.shape(dvf_list[i_dvf])[:-1], dtype=np.bool)
            mask[margin:-margin, margin:-margin, margin:-margin] = True
            if torso_list[i_dvf] is not None:
                mask = mask & torso_list[i_dvf]
            for c in range(len(class_balanced)):
                iclass_address = su.address_generator(setting, 'IClass', data=im_info['data'], deform_exp=im_info['deform_exp'], cn=im_info['cn'],
                                                      type_im=im_info['type_im'], dsmooth=im_info['dsmooth'], c=c, stage=stage)
                if os.path.isfile(iclass_address):
                    i1 = np.load(iclass_address)
                else:
                    if c == 0:
                        # you can add a mask here to prevent selecting pixels twice!
                        i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf_list[i_dvf]) < class_balanced[c]), axis=3)) & mask),
                                                  np.shape(dvf_list[i_dvf])[:-1]).astype(np.int32)
                        # the output of np.where occupy huge part of memory! by converting it to a numpy array lots of memory can be saved!
                    if (c > 0) & (c < (len(class_balanced))):
                        if setting['Dim'] == 2:
                            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
                            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf_list[i_dvf]) < class_balanced[c]), axis=3)) &
                                                               (np.any((np.abs(dvf_list[i_dvf]) >= class_balanced[c - 1]), axis=3)) & mask),
                                                      np.shape(dvf_list[i_dvf])[:-1]).astype(np.int32)
                        if setting['Dim'] == 3:
                            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf_list[i_dvf]) < class_balanced[c]), axis=3)) &
                                                               (np.all((np.abs(dvf_list[i_dvf]) >= class_balanced[c - 1]), axis=3)) & mask),
                                                      np.shape(dvf_list[i_dvf])[:-1]).astype(np.int32)
                    np.save(iclass_address, i1)
                if len(i1) > 0:
                    if len(indices['class'+str(c)]) == 0:
                        indices['class'+str(c)] = np.array(np.c_[i1, i_dvf * np.ones(len(i1), dtype=np.int32)])
                    else:
                        indices['class'+str(c)] = np.concatenate((indices['class'+str(c)], np.array(np.c_[i1, i_dvf * np.ones(len(i1), dtype=np.int32)])), axis=0)
                if setting['verbose']:
                    logging.debug(log_header+': Finding classes done for i = {}, c = {} '.format(i_dvf, c))
        del i1
        end_time = time.time()
        if setting['verbose']:
            logging.debug(log_header+': Searching for {} classes is Done in {:.2f}s'.format(len(class_balanced) + 1, end_time - start_time))
    samples_per_chunk = samples_per_image * len(dvf_list)
    sample_per_chunk_per_class = np.round(samples_per_chunk / (len(class_balanced)))
    number_samples_class = np.empty(len(class_balanced), dtype=np.int32)
    random_state = np.random.RandomState(semi_epoch * 10000 + chunk * 100 + stage)
    selected_indices = np.array([])
    for c, k in enumerate(indices.keys()):
        number_samples_class[c] = min(sample_per_chunk_per_class, np.shape(indices[k])[0])
        # it is possible to have different number in each class. However we perefer to have at least SamplePerChunkPerClass
        if np.shape(indices['class'+str(c)])[0] > 0:
            i1 = random_state.randint(0, high=np.shape(indices['class' + str(c)])[0], size=number_samples_class[c])
            if c == 0 or len(selected_indices) == 0:
                selected_indices = np.concatenate((indices['class' + str(c)][i1, :], c * np.ones([len(i1), 1], dtype=np.int32)), axis=1).astype(np.int32)
            else:
                selected_indices = np.concatenate((selected_indices,
                                                   np.concatenate((indices['class' + str(c)][i1, :],
                                                                   c * np.ones([len(i1), 1], dtype=np.int32)),
                                                                  axis=1)),
                                                  axis=0)
        logging.info(log_header + ': {} of samples in class {} for SemiEpoch = {}, Chunk = {} '.
                     format(number_samples_class[c], c, semi_epoch, chunk))

    if setting['verbose']:
        logging.debug(log_header+': samplesPerChunk is {} for SemiEpoch = {}, Chunk = {} '.format(sum(number_samples_class), semi_epoch, chunk))
    shuffled_index = np.arange(0, len(selected_indices))
    random_state.shuffle(shuffled_index)
    return selected_indices[shuffled_index]


def shuffled_indices_from_chunk_patch_seq(setting, dvf_list=None, torso_list=None, stage_sequence=None, semi_epoch=None,
                                          chunk=None, samples_per_image=None, log_header='', chunk_length_force_to_multiple_of=None):

    margin = setting['Margin']
    class_balanced = setting['ClassBalanced']
    indices = {}
    for c in range(len(class_balanced)):
        indices['class'+str(c)] = []
    start_time = time.time()
    if setting['ParallelSearching']:
        num_cores = multiprocessing.cpu_count() - 2
        results = Parallel(n_jobs=num_cores)(delayed(search_indices_seq)(dvf_label=dvf_list[i], c=c, class_balanced=class_balanced, margin=margin, torso=torso_list[i]['stage1'])
                                             for i in range(len(dvf_list)) for c in range(0, len(class_balanced)))
        for iresults in range(0, len(results)):
            i_dvf = iresults // (len(class_balanced))  # first loop in the Parallel: for i in range(0, len(dvf_list))
            c = iresults % (len(class_balanced))   # second loop in the Parallel: for j in range(0, len(class_balanced)+1)
            if len(results[iresults]):
                if len(indices['class'+str(c)]) == 0:
                    indices['class'+str(c)] = np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])
                else:
                    indices['class'+str(c)] = np.concatenate((indices['class'+str(c)], np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])), axis=0)
        del results
        end_time = time.time()
        if setting['verbose']:
            logging.debug(log_header+' Parallel searching for {} classes is Done in {:.2f}s'.format(len(class_balanced), end_time - start_time))

    samples_per_chunk = samples_per_image * len(dvf_list)
    sample_per_chunk_per_class = np.round(samples_per_chunk / (len(class_balanced)))
    number_samples_class = np.empty(len(class_balanced), dtype=np.int32)
    random_state = np.random.RandomState(semi_epoch * 10000 + chunk * 100 + stage_sequence[0])
    selected_indices = np.array([])
    for c, k in enumerate(indices.keys()):
        number_samples_class[c] = min(sample_per_chunk_per_class * setting['ClassBalancedWeight'][c], np.shape(indices[k])[0])
        # it is possible to have different number in each class. However we perefer to have at least SamplePerChunkPerClass
        if np.shape(indices['class'+str(c)])[0] > 0:
            i1 = random_state.randint(0, high=np.shape(indices['class' + str(c)])[0], size=number_samples_class[c])
            if c == 0 or len(selected_indices) == 0:
                selected_indices = np.concatenate((indices['class' + str(c)][i1, :], c * np.ones([len(i1), 1], dtype=np.int32)), axis=1).astype(np.int32)
            else:
                selected_indices = np.concatenate((selected_indices,
                                                   np.concatenate((indices['class' + str(c)][i1, :],
                                                                   c * np.ones([len(i1), 1], dtype=np.int32)),
                                                                  axis=1)),
                                                  axis=0)
        logging.info(log_header + ': {} of samples in class {} for SemiEpoch = {}, Chunk = {} '.
                     format(number_samples_class[c], c, semi_epoch, chunk))

    if setting['verbose']:
        logging.debug(log_header+': samplesPerChunk is {} for SemiEpoch = {}, Chunk = {} '.format(sum(number_samples_class), semi_epoch, chunk))
    shuffled_index = np.arange(0, len(selected_indices))
    random_state.shuffle(shuffled_index)

    if chunk_length_force_to_multiple_of is not None:
        remainder = len(shuffled_index) % chunk_length_force_to_multiple_of
        if remainder != 0:
            shuffled_index = shuffled_index[0: len(shuffled_index) - remainder]
    return selected_indices[shuffled_index]


def get_ishuffled_folder_write_ishuffled_setting(setting, train_mode, stage, number_of_images_per_chunk,
                                                 samples_per_image, im_info_list_full, full_image,
                                                 chunk_length_force_to_multiple_of=None):
    """
    Thi functions chooses or creates the IShuffledFolder. First it takes a look at the ishuffled_root_folder, if there is no
    folder there it creates the folder and save the ishuffled_setting to a json file.
    If a folder already exists, it compares the ishuffled setting with the json file in that folder. If they are identical, then
    choose that folder. Otherwise it will create another folder by increasing the exp number:

    Example ishuffled_folder: Training_images120_S4_exp0, Training_images120_S4_exp1

    please not that the order of im_lins_info is important. Different order means different images in chunks.

    :return: ishuffled_folder
    """

    ishuffled_setting = {'train_mode': train_mode,
                         'DVFPad': setting['DVFPad_S' + str(stage)],
                         'ImPad': setting['ImPad_S' + str(stage)],
                         'NumberOfImagesPerChunk': number_of_images_per_chunk,
                         'ImInfoList': im_info_list_full,
                         }

    if 'DVFThresholdList' in setting.keys():
        ishuffled_setting['DVFThresholdList'] = copy.deepcopy(setting['DVFThresholdList'])
    if chunk_length_force_to_multiple_of is not None:
        ishuffled_setting['ChunkLengthForceToMultipleOf'] = chunk_length_force_to_multiple_of
    if 'ClassBalancedWeight' in setting.keys():
        ishuffled_setting['ClassBalancedWeight'] = setting['ClassBalancedWeight']

    if not full_image:
        # other important setting in patch based
        ishuffled_setting['ClassBalanced'] = setting['ClassBalanced']
        ishuffled_setting['Margin'] = setting['Margin']
        ishuffled_setting['SamplePerImage'] = samples_per_image


    ishuffled_folder = None
    ishuffled_exp = 0
    folder_found = False
    while not folder_found:
        ishuffled_folder = su.address_generator(setting, 'IShuffledFolder',
                                                train_mode=train_mode,
                                                stage=stage,
                                                ishuffled_exp=ishuffled_exp,
                                                im_list_info=im_info_list_full)
        ishuffled_setting_address = su.address_generator(setting, 'IShuffledSetting',
                                                         train_mode=train_mode,
                                                         stage=stage,
                                                         ishuffled_exp=ishuffled_exp,
                                                         im_list_info=im_info_list_full)
        if not (os.path.isdir(ishuffled_folder)):
            os.makedirs(ishuffled_folder)
            with open(ishuffled_setting_address, 'w') as f:
                f.write(json.dumps(ishuffled_setting, sort_keys=True, indent=4, separators=(',', ': ')))
            folder_found = True
        else:
            with open(ishuffled_setting_address, 'r') as f:
                ishuffled_setting_exp = json.load(f)
            if ishuffled_setting_exp == ishuffled_setting:
                folder_found = True
            else:
                ishuffled_exp = ishuffled_exp + 1
    return ishuffled_folder


def extract_batch(setting, stage, fixed_im_list, deformed_im_list, dvf_list, ish,
                  batch_counter, batch_size, end_batch, full_image):
    if full_image:
        batch_both, batch_dvf = extract_batch_from_image(setting, stage, fixed_im_list, deformed_im_list,
                                                         dvf_list, ish, batch_counter, batch_size, end_batch)
    else:
        batch_both, batch_dvf = extract_batch_from_patch(setting, stage, fixed_im_list, deformed_im_list,
                                                         dvf_list, ish, batch_counter, batch_size, end_batch)
    return batch_both, batch_dvf


def extract_batch_seq(setting, stage_sequence, fixed_im_list, moved_im_list, dvf_list, ish,
                      batch_counter, batch_size, end_batch, full_image):
    if full_image:
        print('not implemented')
    else:
        batch_both, batch_dvf = extract_batch_from_patch_seq(setting, stage_sequence, fixed_im_list, moved_im_list,
                                                             dvf_list, ish, batch_counter, batch_size, end_batch)
    return batch_both, batch_dvf

def extract_batch_from_image(setting, stage, fixed_im_list, deformed_im_list, dvf_list, ish,
                             batch_counter, batch_size, end_batch):
    batch_im = np.stack([fixed_im_list[ish[i]] for i in range(batch_counter * batch_size, end_batch)], axis=0)
    batch_deformed = np.stack([deformed_im_list[ish[i]] for i in range(batch_counter * batch_size, end_batch)], axis=0)
    batch_dvf = np.stack([dvf_list[ish[i]] for i in range(batch_counter * batch_size, end_batch)], axis=0)
    batch_both = np.stack((batch_im, batch_deformed), axis=setting['Dim']+1)
    return batch_both, batch_dvf


def extract_batch_from_patch(setting, stage, fixed_im_list, deformed_im_list, dvf_list, ish,
                             batch_counter, batch_size, end_batch):
    # ish [: , 0] the index of the sample that is gotten from np.where
    # ish [: , 1] the the number of the image in FixedImList
    # ish [: , 2] the the number of class, which is not needed anymore!!
    r = setting['R']
    ry = setting['Ry']

    if setting['Dim'] == 2:
        shift_center = setting['ImPad_S' + str(stage)] - setting['DVFPad_S' + str(stage)]
        batch_im = np.stack([fixed_im_list[ish[i, 1]][
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:2])[0],
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:2])[1] - r + shift_center:
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:2])[1] + r + shift_center + 1,
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:2])[2] - r + shift_center:
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:2])[2] + r + shift_center + 1,
                             np.newaxis] for i in range(batch_counter * batch_size, end_batch)])
        batch_deformed = np.stack([deformed_im_list[ish[i, 1]][
                                   np.unravel_index(ish[i, 0], np.shape(deformed_im_list[ish[i, 1]])[0:2])[0],
                                   np.unravel_index(ish[i, 0], np.shape(deformed_im_list[ish[i, 1]])[0:2])[1] - r + shift_center:
                                   np.unravel_index(ish[i, 0], np.shape(deformed_im_list[ish[i, 1]])[0:2])[1] + r + shift_center + 1,
                                   np.unravel_index(ish[i, 0], np.shape(deformed_im_list[ish[i, 1]])[0:2])[2] - r + shift_center:
                                   np.unravel_index(ish[i, 0], np.shape(deformed_im_list[ish[i, 1]])[0:2])[2] + r + shift_center + 1,
                                   np.newaxis] for i in range(batch_counter * batch_size, end_batch)])
        batch_both = np.concatenate((batch_im, batch_deformed), axis=3)
        batch_dvf = np.stack([dvf_list[ish[i, 1]][
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))[0],
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))[1] - ry:
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))[1] + ry + 1,
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))[2] - ry:
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))[2] + ry + 1,
                              0:2] for i in range(batch_counter * batch_size, end_batch)])
    elif setting['Dim'] == 3:
        shift_center = setting['ImPad_S' + str(stage)] - setting['DVFPad_S' + str(stage)]
        batch_im = np.stack([fixed_im_list[ish[i, 1]][
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] - r + shift_center:
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] + r + shift_center + 1,
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] - r + shift_center:
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] + r + shift_center + 1,
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] - r + shift_center:
                             np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] + r + shift_center + 1,
                             np.newaxis] for i in range(batch_counter * batch_size, end_batch)])
        batch_deformed = np.stack([deformed_im_list[ish[i, 1]][
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] - r + shift_center:
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] + r + shift_center + 1,
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] - r + shift_center:
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] + r + shift_center + 1,
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] - r + shift_center:
                                   np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] + r + shift_center + 1,
                                   np.newaxis] for i in range(batch_counter * batch_size, end_batch)])
        batch_both = np.concatenate((batch_im, batch_deformed), axis=4)
        batch_dvf = np.stack([dvf_list[ish[i, 1]][
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] - ry:
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[0] + ry + 1,
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] - ry:
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[1] + ry + 1,
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] - ry:
                              np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]])[0:3])[2] + ry + 1,
                              0:3] for i in range(batch_counter * batch_size, end_batch)])
    else:
        raise ValueError('Dim should be either 2D or 3D')
    return batch_both, batch_dvf


def extract_batch_from_patch_seq(setting, stage_sequence, fixed_im_list, moved_im_list, dvf_list, ish,
                                 batch_counter, batch_size, end_batch):
    # ish [: , 0] the index of the sample that is gotten from np.where
    # ish [: , 1] the the number of the image in fixed_im_list
    # ish [: , 2] the the number of class, which is not needed anymore!!
    r = setting['R']
    ry = setting['Ry']


    # ind = dict()
    # ind['s1_no_padding'] = [list(np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))) for i in range(batch_counter * batch_size, end_batch)]
    # for stage in stage_sequence:
    #     if stage != 1:
    #         ind['s'+str(stage)+'_no_padding'] = [[None, None, None]  for _ in range(len(ind['s1_no_padding']))]
    #         ind['s'+str(stage)+'_extra_shift'] = [[None, None, None]  for _ in range(len(ind['s1_no_padding']))]
    #         for i_index, index_s1_no_padding in enumerate(ind['s1_no_padding']):
    #             ind['s'+str(stage)+'_no_padding'][i_index] = [index_s1_no_padding[0]//stage+1, index_s1_no_padding[1]//stage+1, index_s1_no_padding[2]//stage+1]
    #             for ind in range(3):
    #                 if ind['s'+str(stage)+'_no_padding'][i_index][ind] < ry:
    #                     ind['s'+str(stage)+'_extra_before'][i_index][ind] = ry - ind['s'+str(stage)+'_no_padding'][i_index][ind]
    #                     ind['s'+str(stage)+'_no_padding'][i_index][ind] = ry

    ind = dict()
    ind['S1_no_padding'] = np.array([list(np.unravel_index(ish[i, 0], np.shape(dvf_list[ish[i, 1]]))) for i in range(batch_counter * batch_size, end_batch)], dtype=np.int16)
    ind['dvf_size_no_padding'] = np.array([np.shape(dvf_list[ish[i, 1]]) for i in range(batch_counter * batch_size, end_batch)])
    # why it cannot work?????????
    # for stage in stage_sequence:
    #     ind['s'+str(stage)+'_fixed_im_size'] = np.zeros(ind['s1_no_padding'].shape, dtype=np.int16)
    #     ind['s'+str(stage)+'_fixed_im_size'][:] = np.array(copy.deepcopy([list(np.shape(fixed_im_list[ish[i, 1]]['stage'+str(stage)])) for i in range(batch_counter * batch_size, end_batch)]))[:]

    for stage in stage_sequence:
        if stage != 1:
            ind['S'+str(stage)+'_no_padding'] = np.zeros(ind['S1_no_padding'].shape, dtype=np.int16)
            ind['S'+str(stage)+'_no_padding'] = np.zeros(ind['S1_no_padding'].shape, dtype=np.int16)
            # ind['s'+str(stage)+'_extra_before'] = np.zeros(ind['s1_no_padding'].shape, dtype=np.int16)
            # ind['s'+str(stage)+'_extra_after'] = np.zeros(ind['s1_no_padding'].shape, dtype=np.int16)
            for patch_i in range(ind['S1_no_padding'].shape[0]):
                ind['S'+str(stage)+'_no_padding'][patch_i, :] = np.round(ind['S1_no_padding'][patch_i, :]/stage)

        ind['S'+str(stage)] = ind['S'+str(stage)+'_no_padding'] + setting['ImPad_S'+str(stage)]

            #     for ax_i in range(3):
            #         if ind['s'+str(stage)+'_no_padding'][patch_i][ax_i] < ry:
            #             ind['s'+str(stage)+'_extra_before'][patch_i][ax_i] = ry - ind['s'+str(stage)+'_no_padding'][patch_i][ax_i]
            #             # ind['s'+str(stage)+'_no_padding'][patch_i][ax_i] = ry
            #         shape_im_no_padding = np.shape(fixed_im_list[ish[patch_i, 1]]['stage'+str(stage)])[ax_i] - setting['ImPad_S1']
            #         if ind['s'+str(stage)+'_no_padding'][patch_i][ax_i] > shape_im_no_padding - ry - 1:
            #             ind['s'+str(stage)+'_extra_after'][patch_i][ax_i] = ind['s'+str(stage)+'_no_padding'][patch_i][ax_i] - (shape_im_no_padding - ry - 1)
            #             ind['s'+str(stage)+'_no_padding'][patch_i][ax_i] = shape_im_no_padding - ry -1

    #
    # batch_both = dict()
    # for stage in stage_sequence:
    #     batch_fixed = np.stack([fixed_im_list[ish[i, 1]]['stage'+str(stage)][
    #                             ind['S'+str(stage)][i][0] - r: ind['S'+str(stage)][i][0] + r + 1,
    #                             ind['S'+str(stage)][i][1] - r: ind['S'+str(stage)][i][1] + r + 1,
    #                             ind['S'+str(stage)][i][2] - r: ind['S'+str(stage)][i][2] + r + 1,
    #                             np.newaxis] for i in range(batch_counter * batch_size, end_batch)]).copy()
    #     batch_moved = np.stack([moved_im_list[ish[i, 1]]['stage'+str(stage)][
    #                             ind['S' + str(stage)][i][0] - r: ind['S' + str(stage)][i][0] + r + 1,
    #                             ind['S' + str(stage)][i][1] - r: ind['S' + str(stage)][i][1] + r + 1,
    #                             ind['S' + str(stage)][i][2] - r: ind['S' + str(stage)][i][2] + r + 1,
    #                             np.newaxis] for i in range(batch_counter * batch_size, end_batch)]).copy()
    #     batch_both_s = np.concatenate((batch_fixed.copy(), batch_moved.copy()), axis=4)
    #     batch_both['stage'+str(stage)] = copy.deepcopy(batch_both_s)


    batch_both = dict()
    batch_fixed_s1 = np.stack([fixed_im_list[ish[i, 1]]['stage1'][
                            ind['S1'][i_batch][0] - r: ind['S1'][i_batch][0] + r + 1,
                            ind['S1'][i_batch][1] - r: ind['S1'][i_batch][1] + r + 1,
                            ind['S1'][i_batch][2] - r: ind['S1'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_moved_s1 = np.stack([moved_im_list[ish[i, 1]]['stage1'][
                            ind['S1'][i_batch][0] - r: ind['S1'][i_batch][0] + r + 1,
                            ind['S1'][i_batch][1] - r: ind['S1'][i_batch][1] + r + 1,
                            ind['S1'][i_batch][2] - r: ind['S1'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_both_s1 = np.concatenate((batch_fixed_s1, batch_moved_s1), axis=4)
    batch_both['stage1'] = batch_both_s1

    batch_fixed_s2 = np.stack([fixed_im_list[ish[i, 1]]['stage2'][
                            ind['S2'][i_batch][0] - r: ind['S2'][i_batch][0] + r + 1,
                            ind['S2'][i_batch][1] - r: ind['S2'][i_batch][1] + r + 1,
                            ind['S2'][i_batch][2] - r: ind['S2'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_moved_s2 = np.stack([moved_im_list[ish[i, 1]]['stage2'][
                            ind['S2'][i_batch][0] - r: ind['S2'][i_batch][0] + r + 1,
                            ind['S2'][i_batch][1] - r: ind['S2'][i_batch][1] + r + 1,
                            ind['S2'][i_batch][2] - r: ind['S2'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_both_s2 = np.concatenate((batch_fixed_s2, batch_moved_s2), axis=4)
    batch_both['stage2'] = batch_both_s2

    batch_fixed_s4 = np.stack([fixed_im_list[ish[i, 1]]['stage4'][
                            ind['S4'][i_batch][0] - r: ind['S4'][i_batch][0] + r + 1,
                            ind['S4'][i_batch][1] - r: ind['S4'][i_batch][1] + r + 1,
                            ind['S4'][i_batch][2] - r: ind['S4'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_moved_s4 = np.stack([moved_im_list[ish[i, 1]]['stage4'][
                            ind['S4'][i_batch][0] - r: ind['S4'][i_batch][0] + r + 1,
                            ind['S4'][i_batch][1] - r: ind['S4'][i_batch][1] + r + 1,
                            ind['S4'][i_batch][2] - r: ind['S4'][i_batch][2] + r + 1,
                            np.newaxis] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])
    batch_both_s4 = np.concatenate((batch_fixed_s4, batch_moved_s4), axis=4)
    batch_both['stage4'] = batch_both_s4

    batch_dvf_mat = np.stack([dvf_list[ish[i, 1]][
                          ind['S1_no_padding'][i_batch, 0] - ry: ind['S1_no_padding'][i_batch, 0] + ry + 1,
                          ind['S1_no_padding'][i_batch, 1] - ry: ind['S1_no_padding'][i_batch, 1] + ry + 1,
                          ind['S1_no_padding'][i_batch, 2] - ry: ind['S1_no_padding'][i_batch, 2] + ry + 1,
                          ] for i_batch, i in enumerate(range(batch_counter * batch_size, end_batch))])

    batch_dvf = batch_dvf_to_one_hot(setting, batch_dvf_mat)

    # import PIL
    # batch_pil = 0
    # slice_pil = 50
    # img1 = PIL.Image.fromarray(fixed_im_list['stage1'][batch_pil, slice_pil, :, :, 0])
    # img1.show()
    # img1 = PIL.Image.fromarray(batch_both['stage2'][batch_pil, slice_pil, :, :, 0])
    # img1.show()
    #
    #
    #
    # img1 = PIL.Image.fromarray(batch_both['stage1'][batch_pil, slice_pil, :, :, 0])
    # img1.show()
    # img1 = PIL.Image.fromarray(batch_both['stage2'][batch_pil, slice_pil, :, :, 0])
    # img1.show()
    # img1 = PIL.Image.fromarray(batch_both['stage4'][batch_pil, slice_pil, :, :, 0])
    # img1.show()
    #
    #
    #
    # img2 = PIL.Image.fromarray(batch_both['stage2'][1, slice_pil, :, :, 0])
    # img2.show()
    # img3 = PIL.Image.fromarray(batch_both['stage2'][2, slice_pil, :, :, 0])
    # img3.show()

    return batch_both, batch_dvf


def batch_dvf_to_one_hot(setting, batch_dvf):
    """
    please improve it later
    :param setting:
    :param batch_dvf:
    :return:
    """
    number_of_labels = setting['NumberOfLabels']
    batch_dvf_one_hot_t0 = np.zeros(np.shape(batch_dvf) + (number_of_labels,), dtype=np.int8)
    batch_dvf_one_hot_t1 = np.zeros(np.shape(batch_dvf) + (number_of_labels,), dtype=np.int8)
    batch_dvf_one_hot_t2 = np.zeros(np.shape(batch_dvf) + (number_of_labels,), dtype=np.int8)

    # i1 = np.where(batch_dvf == 0)
    # i2 = i1 + (np.zeros(np.shape(i1[0]), dtype=np.int),)
    # batch_dvf_one_hot_t0[i2] = 1
    # batch_dvf_one_hot_t1[i2] = 1
    # batch_dvf_one_hot_t2[i2] = 1

    # batch_dvf = batch_dvf - 1  # dvf was saved from 1 to 7, now I like to change the classes from 0 to 6
    # batch_dvf[batch_dvf == -1] = 0

    if not setting['AugmentedLabel']:
        indices_label = [None for _ in range(number_of_labels)]
        for label in setting['Labels_time2']:
            indices_label[label] = np.where(batch_dvf == label)
            i2 = indices_label[label] + (np.ones(np.shape(indices_label[label][0]), dtype=np.int) * label,)
            batch_dvf_one_hot_t2[i2] = 1

        for label in setting['Labels_time2']:
            if label in setting['Labels_time0_group']:
                i2 = indices_label[label] + (np.ones(np.shape(indices_label[label][0]), dtype=np.int) * label,)
                batch_dvf_one_hot_t0[i2] = 1
            else:
                label_times_list_elements = filter(check_list, setting['Labels_time0_group'])
                label_times_list_elements = [i for i in label_times_list_elements]
                for list_element in label_times_list_elements:
                    if label in list_element:
                        for label_in_element in list_element:
                            i2 = indices_label[label] + (np.ones(np.shape(indices_label[label][0]), dtype=np.int) * label_in_element,)
                            batch_dvf_one_hot_t0[i2] = 1

            if label in setting['Labels_time1_group']:
                i2 = indices_label[label] + (np.ones(np.shape(indices_label[label][0]), dtype=np.int) * label,)
                batch_dvf_one_hot_t1[i2] = 1
            else:
                label_times_list_elements = filter(check_list, setting['Labels_time1_group'])
                label_times_list_elements = [i for i in label_times_list_elements]
                for list_element in label_times_list_elements:
                    if label in list_element:
                        for label_in_element in list_element:
                            i2 = indices_label[label] + (np.ones(np.shape(indices_label[label][0]), dtype=np.int) * label_in_element,)
                            batch_dvf_one_hot_t1[i2] = 1

    if setting['AugmentedLabel']:
        for label in setting['Labels_time2']:
            i1 = np.where(batch_dvf == label)
            i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
            batch_dvf_one_hot_t2[i2] = 1

        for label in setting['Labels_time1']:
            # if label != 0:
            if label in setting['LabelsDefinition'].keys():
                if len(setting['LabelsDefinition'][label]) == 2:
                    i1 = np.where(np.logical_or(batch_dvf == setting['LabelsDefinition'][label][0], batch_dvf == setting['LabelsDefinition'][label][1]))
                    i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
                elif len(setting['LabelsDefinition'][label]) == 3:
                    i1 = np.where(np.logical_or(batch_dvf == setting['LabelsDefinition'][label][0],
                                  batch_dvf == setting['LabelsDefinition'][label][1],
                                  batch_dvf == setting['LabelsDefinition'][label][2]))
                    i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
                else:
                    raise ValueError('not defined for more than 3')
            else:
                i1 = np.where(batch_dvf == label)
                i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
            batch_dvf_one_hot_t1[i2] = 1

        for label in setting['Labels_time0']:
            # if label != 0:
            if label in setting['LabelsDefinition'].keys():
                if len(setting['LabelsDefinition'][label]) == 2:
                    i1 = np.where(np.logical_or(batch_dvf == setting['LabelsDefinition'][label][0], batch_dvf == setting['LabelsDefinition'][label][1]))
                    i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
                elif len(setting['LabelsDefinition'][label]) == 3:
                    i1 = np.where(np.logical_or(batch_dvf == setting['LabelsDefinition'][label][0],
                                  batch_dvf == setting['LabelsDefinition'][label][1],
                                  batch_dvf == setting['LabelsDefinition'][label][2]))
                    i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
                else:
                    raise ValueError('not defined for more than 3')
            else:
                i1 = np.where(batch_dvf == label)
                i2 = i1 + (np.ones(np.shape(i1[0]), dtype=np.int) * label,)
            batch_dvf_one_hot_t0[i2] = 1

    batch_dvf_one_hot = {'t0': batch_dvf_one_hot_t0,
                         't1': batch_dvf_one_hot_t1,
                         't2': batch_dvf_one_hot_t2,}
    # c = 16
    # i = 8
    # print('dvf mag is {}'.format(batch_dvf[i, c, c, c]))
    # print('one hot t0 {}'.format(batch_dvf_one_hot_t0[i, c, c, c, :]))
    # print('one hot t1 {}'.format(batch_dvf_one_hot_t1[i, c, c, c, :]))
    # print('one hot t2 {}'.format(batch_dvf_one_hot_t2[i, c, c, c, :]))


    return batch_dvf_one_hot


def select_im_from_semiepoch(setting, im_info_list_full=None, semi_epoch=None, chunk=None, number_of_images_per_chunk=None):
    im_info_list_full = copy.deepcopy(im_info_list_full)
    random_state = np.random.RandomState(semi_epoch)
    if setting['Randomness']:
        random_indices = random_state.permutation(len(im_info_list_full))
    else:
        random_indices = np.arange(len(im_info_list_full))
    lower_range = chunk * number_of_images_per_chunk
    upper_range = (chunk + 1) * number_of_images_per_chunk
    if upper_range >= len(im_info_list_full):
        upper_range = len(im_info_list_full)

    indices_chunk = random_indices[lower_range: upper_range]
    im_info_list = [im_info_list_full[i] for i in indices_chunk]
    return im_info_list


def check_list(input_element):
    if isinstance(input_element, list):
        return True
    else:
        return False
