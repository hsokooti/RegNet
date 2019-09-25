import copy
import logging
import numpy as np
import os
from . import utils as reading_utils
import functions.setting.setting_utils as su
import functions.artificial_generation.dvf_generation as synth


class Images(object):
    def __init__(self,
                 setting=None,
                 number_of_images_per_chunk=None,  # number of images that I would like to load in RAM
                 samples_per_image=None,
                 im_info_list_full=None,
                 train_mode=None,
                 semi_epoch=0,
                 stage=None,
                 ):
        self._setting = setting
        self._number_of_images_per_chunk = number_of_images_per_chunk
        self._samples_per_image = samples_per_image
        self._chunk = 0
        self._chunks_completed = 0
        self._semi_epochs_completed = 0
        self._semi_epoch = semi_epoch
        self._batch_counter = 0
        self._fixed_im_list = [None] * number_of_images_per_chunk
        self._deformed_im_list = [None] * number_of_images_per_chunk
        self._dvf_list = [None] * number_of_images_per_chunk
        self._im_info_list_full = im_info_list_full
        self._train_mode = train_mode
        if stage is None:
            stage = setting['stage']
        self._stage = stage
        if self._train_mode == 'Training':
            self._mode_synthetic_dvf = 'reading'
        if self._train_mode == 'Validation':
            self._mode_synthetic_dvf = 'generation'
        self._ishuffled = None
        ishuffled_folder = su.address_generator(setting, 'IShuffledFolder', train_mode=train_mode, stage=stage)
        if not (os.path.isdir(ishuffled_folder)):
            os.makedirs(ishuffled_folder)

    def fill(self):
        number_of_images_per_chunk = self._number_of_images_per_chunk
        if self._train_mode == 'Training':
            # Make all lists empty in training mode. In the validation or test mode, we keep the same chunk forever. So no need to make it empty and refill it again.
            self._fixed_im_list = [None] * number_of_images_per_chunk
            self._deformed_im_list = [None] * number_of_images_per_chunk
            self._dvf_list = [None] * number_of_images_per_chunk
        
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0

        im_info_list_full = copy.deepcopy(self._im_info_list_full)
        np.random.seed(self._semi_epoch)
        if self._setting['Randomness']:
            random_indices = np.random.permutation(len(im_info_list_full))
        else:
            random_indices = np.arange(len(im_info_list_full))

        lower_range = self._chunk * number_of_images_per_chunk
        upper_range = (self._chunk+1) * number_of_images_per_chunk
        if upper_range >= len(im_info_list_full):
            upper_range = len(im_info_list_full)
            self._semi_epochs_completed = 1
            number_of_images_per_chunk = upper_range - lower_range  # In cases when last chunk of images are smaller than the self._numberOfImagesPerChunk
            self._fixed_im_list = [None] * number_of_images_per_chunk
            self._deformed_im_list = [None] * number_of_images_per_chunk
            self._dvf_list = [None] * number_of_images_per_chunk

        torso_list = [None] * len(self._dvf_list)
        indices_chunk = random_indices[lower_range: upper_range]
        im_info_list = [im_info_list_full[i] for i in indices_chunk]
        for i_index_im, index_im in enumerate(indices_chunk):
            self._fixed_im_list[i_index_im], self._deformed_im_list[i_index_im], self._dvf_list[i_index_im], torso_list[i_index_im] = \
                synth.get_dvf_and_deformed_images(self._setting,
                                                  im_info=im_info_list_full[index_im],
                                                  stage=self._stage,
                                                  mode_synthetic_dvf=self._mode_synthetic_dvf
                                                  )
            if self._setting['verbose']:
                logging.debug('direct: Data='+im_info_list_full[index_im]['data'] +
                              ', TypeIm={}, CN={}, Dsmooth={}, stage={} is loaded'.format(im_info_list_full[index_im]['type_im'], im_info_list_full[index_im]['cn'],
                                                                                          im_info_list_full[index_im]['dsmooth'], self._stage))
        ishuffled_address = su.address_generator(self._setting, 'IShuffled', train_mode=self._train_mode, number_of_images_per_chunk=self._number_of_images_per_chunk,
                                                 samples_per_image=self._samples_per_image, semi_epoch=self._semi_epoch, chunk=self._chunk, stage=self._stage)
        if os.path.isfile(ishuffled_address):
            self._ishuffled = np.load(ishuffled_address)
        else:
            self._ishuffled = reading_utils.shuffled_indices_from_chunk(self._setting, dvf_list=self._dvf_list, torso_list=torso_list, im_info_list=im_info_list,
                                                                        stage=self._stage, semi_epoch=self._semi_epoch, chunk=self._chunk, samples_per_image=self._samples_per_image,
                                                                        number_of_images_per_chunk=number_of_images_per_chunk, log_header='direct')
            np.save(ishuffled_address, self._ishuffled)

    def go_to_next_chunk(self):
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0
        else:
            self._chunk = self._chunk + 1
        logging.debug('direct: NextChunk, dataMode=' + self._train_mode + ', semiEpoch={}, Chunk={}, batchCounter={} '.format(self._semi_epoch, self._chunk, self._batch_counter))

    def next_batch(self, batch_size):
        if self._chunks_completed:
            self._chunk = self._chunk + 1
            self.fill()
            self._batch_counter = 0
            self._chunks_completed = 0
        ish = self._ishuffled
        end_batch = (self._batch_counter + 1) * batch_size
        if end_batch >= len(ish):
            self._chunks_completed = 1
            end_batch = len(ish)
        batch_both, batch_dvf = reading_utils.extract_batch(self._setting, stage=self._stage, ish=ish,
                                                            fixed_im_list=self._fixed_im_list,
                                                            deformed_im_list=self._deformed_im_list,
                                                            dvf_list=self._dvf_list,
                                                            batch_counter=self._batch_counter, batch_size=batch_size, end_batch=end_batch)
        if self._setting['verbose']:
            logging.debug('direct: dataMode='+self._train_mode+', semiEpoch={}, Chunk={}, batchCounter={} , endBatch={} '.format(self._semi_epoch, self._chunk, self._batch_counter, end_batch))
        self._batch_counter = self._batch_counter + 1
        return batch_both, batch_dvf

    def reset_validation(self):
        self._batch_counter = 0
        self._chunks_completed = 0
        self._chunk = 0
        self._semi_epochs_completed = 0
        self._semi_epoch = 0

    def copy_from_thread(self, reading_thread):
        self._setting = copy.deepcopy(reading_thread._setting)
        self._number_of_images_per_chunk = copy.copy(reading_thread._number_of_images_per_chunk)
        self._samples_per_image = copy.copy(reading_thread._samples_per_image)
        self._batch_counter = 0
        self._chunk = copy.copy(reading_thread._chunk)
        self._chunks_completed = copy.copy(reading_thread._chunks_completed)
        self._semi_epochs_completed = copy.copy(reading_thread._semi_epochs_completed)
        self._semi_epoch = copy.copy(reading_thread._semi_epoch)
        self._fixed_im_list = copy.deepcopy(reading_thread._fixed_im_list)
        self._deformed_im_list = copy.deepcopy(reading_thread._deformed_im_list)
        self._dvf_list = copy.deepcopy(reading_thread._dvf_list)
        self._im_info_list_full = copy.deepcopy(reading_thread._im_info_list_full)
        self._train_mode = copy.copy(reading_thread._train_mode)
        self._ishuffled = copy.deepcopy(reading_thread._ishuffled)
        self._stage = copy.copy(self._stage)
