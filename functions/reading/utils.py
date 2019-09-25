import copy
import functions.setting.setting_utils as su
from joblib import Parallel, delayed
import logging
import multiprocessing
import numpy as np
import os
import time


def search_indices(dvf, c, class_balanced, margin, dim_im, torso):
    """
    This function searches for voxels based on the classBalanced in the parallel mode: if Setting['ParallelProcessing'] == True

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
        if dim_im == '2D':
            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf) < class_balanced[c]), axis=3)) & (np.any((np.abs(dvf) >= class_balanced[c - 1]), axis=3)) &
                                               mask), np.shape(dvf)[:-1]).astype(np.int32)
        elif dim_im == '3D':
            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf) < class_balanced[c]), axis=3)) & (np.all((np.abs(dvf) >= class_balanced[c - 1]), axis=3)) &
                                               mask), np.shape(dvf)[:-1]).astype(np.int32)
    return i1


def shuffled_indices_from_chunk(setting, dvf_list=None, torso_list=None, im_info_list=None, stage=None, semi_epoch=None,
                                chunk=None, samples_per_image=None, number_of_images_per_chunk=None, log_header=''):
    for single_dict in setting['DataExpDict']:
        iclass_folder = su.address_generator(setting, 'IClassFolder', data=single_dict['data'], deform_exp=single_dict['deform_exp'], stage=stage)
        if not (os.path.isdir(iclass_folder)):
            os.makedirs(iclass_folder)

    margin = setting['Margin']
    class_balanced = setting['classBalanced']
    indices = {}
    start_time = time.time()
    if setting['ParallelProcessing']:
        num_cores = multiprocessing.cpu_count() - 2
        results = [None] * len(dvf_list) * len(class_balanced)
        count_iclass_loaded = 0
        for i_dvf, im_info in enumerate(im_info_list):
            for c in range(0, len(class_balanced)):
                iclass_address = su.address_generator(setting, 'IClass', data=im_info['data'], deform_exp=im_info['deform_exp'], cn=im_info['cn'],
                                                      type_im=im_info['type_im'], dsmooth=im_info['dsmooth'], c=c, stage=stage)
                if os.path.isfile(iclass_address):
                    results[i_dvf * len(class_balanced) + c] = np.load(iclass_address)  # double checked
                    count_iclass_loaded += 1
        if count_iclass_loaded != len(results):
            logging.debug(log_header+': not all I1 found. start calculating... semiEpoch = {}, Chunk = {}, stage={}'.format(semi_epoch, chunk, stage))
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
            if (i_dvf == 0) or (len(indices['class' + str(c)]) == 0):
                indices['class' + str(c)] = np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])
            else:
                indices['class' + str(c)] = np.concatenate((indices['class' + str(c)], np.array(np.c_[results[iresults], i_dvf * np.ones(len(results[iresults]), dtype=np.int32)])), axis=0)
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
            for c in range(0, len(class_balanced)):
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
                        if setting['Dim'] == '2D':
                            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
                            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf_list[i_dvf]) < class_balanced[c]), axis=3)) &
                                                               (np.any((np.abs(dvf_list[i_dvf]) >= class_balanced[c - 1]), axis=3)) & mask),
                                                      np.shape(dvf_list[i_dvf])[:-1]).astype(np.int32)
                        if setting['Dim'] == '3D':
                            i1 = np.ravel_multi_index(np.where((np.all((np.abs(dvf_list[i_dvf]) < class_balanced[c]), axis=3)) &
                                                               (np.all((np.abs(dvf_list[i_dvf]) >= class_balanced[c - 1]), axis=3)) & mask),
                                                      np.shape(dvf_list[i_dvf])[:-1]).astype(np.int32)
                    np.save(iclass_address, i1)
                if (i_dvf == 0) or (len(indices['class' + str(c)]) == 0):
                    indices['class' + str(c)] = np.array(np.c_[i1, i_dvf * np.ones(len(i1), dtype=np.int32)])
                else:
                    indices['class' + str(c)] = np.concatenate((indices['class' + str(c)], np.array(np.c_[i1, i_dvf * np.ones(len(i1), dtype=np.int32)])), axis=0)
                if setting['verbose']:
                    logging.debug(log_header+': Finding classes done for i = {}, c = {} '.format(i_dvf, c))
        del i1
        end_time = time.time()
        if setting['verbose']:
            logging.debug(log_header+': Searching for {} classes is Done in {:.2f}s'.format(len(class_balanced) + 1, end_time - start_time))
    samples_per_chunk = samples_per_image * number_of_images_per_chunk
    sample_per_chunk_per_class = np.round(samples_per_chunk / (len(class_balanced)))
    number_samples_class = np.empty(len(class_balanced), dtype=np.int32)
    np.random.seed(semi_epoch * 10000 + chunk * 100 + stage)
    selected_indices = np.array([])
    for c, k in enumerate(indices.keys()):
        number_samples_class[c] = min(sample_per_chunk_per_class, np.shape(indices[k])[0])
        # it is possible to have different number in each class. However we perefer to have at least SamplePerChunkPerClass
        if np.shape(indices['class' + str(c)])[0] > 0:
            i1 = np.random.randint(0, high=np.shape(indices['class' + str(c)])[0], size=number_samples_class[c])
            if c == 0 or len(selected_indices) == 0:
                selected_indices = np.concatenate((indices['class' + str(c)][i1, :], c * np.ones([len(i1), 1], dtype=np.int32)), axis=1).astype(np.int32)
            else:
                selected_indices = np.concatenate((selected_indices, np.concatenate((indices['class' + str(c)][i1, :], c * np.ones([len(i1), 1], dtype=np.int32)), axis=1)), axis=0)
        else:
            logging.info(log_header+': no samples in class {} for semiEpoch = {}, Chunk = {} '.format(c, semi_epoch, chunk))
    if setting['verbose']:
        logging.debug(log_header+': samplesPerChunk is {} for semiEpoch = {}, Chunk = {} '.format(sum(number_samples_class), semi_epoch, chunk))
    shuffled_index = np.arange(0, len(selected_indices))
    np.random.shuffle(shuffled_index)
    return selected_indices[shuffled_index]


def extract_batch(setting, stage=None, fixed_im_list=None, deformed_im_list=None, dvf_list=None, ish=None,
                  batch_counter=None, batch_size=None, end_batch=None):
    # ish [: , 0] the index of the sample that is gotten from np.where
    # ish [: , 1] the the number of the image in FixedImList
    # ish [: , 2] the the number of class, which is not needed anymore!!
    r = setting['R']
    ry = setting['Ry']

    if setting['Dim'] == '2D':
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
    elif setting['Dim'] == '3D':
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


def select_im_from_semiepoch(setting, im_info_list_full=None, semi_epoch=None, chunk=None, number_of_images_per_chunk=None):
    im_info_list_full = copy.deepcopy(im_info_list_full)
    np.random.seed(semi_epoch)
    if setting['Randomness']:
        random_indices = np.random.permutation(len(im_info_list_full))
    else:
        random_indices = np.arange(len(im_info_list_full))
    lower_range = chunk * number_of_images_per_chunk
    upper_range = (chunk + 1) * number_of_images_per_chunk
    if upper_range >= len(im_info_list_full):
        upper_range = len(im_info_list_full)

    indices_chunk = random_indices[lower_range: upper_range]
    im_info_list = [im_info_list_full[i] for i in indices_chunk]
    return im_info_list
