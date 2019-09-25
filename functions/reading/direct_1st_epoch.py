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
        self._dvf_list = [None] * number_of_images_per_chunk
        self._im_info_list_full = im_info_list_full
        self._train_mode = train_mode
        if stage is None:
            stage = setting['stage']
        self._stage = stage
        self._ishuffled = None
        ishuffled_folder = su.address_generator(setting, 'IShuffledFolder', train_mode=train_mode, stage=stage)
        if not (os.path.isdir(ishuffled_folder)):
            os.makedirs(ishuffled_folder)

    def run(self):
        while self._semi_epoch == 0:
            ishuffled_address = su.address_generator(self._setting, 'IShuffled', train_mode=self._train_mode, number_of_images_per_chunk=self._number_of_images_per_chunk,
                                                     samples_per_image=self._samples_per_image, semi_epoch=self._semi_epoch, chunk=self._chunk, stage=self._stage)
            while os.path.isfile(ishuffled_address) and not self._semi_epoch:
                logging.debug('Direct1stEpoch: for stage={}, semiEpoch={}, Chunk={} is already generated, going to next chunk'.format(self._semi_epoch, self._stage, self._chunk))
                self.go_to_next_chunk_without_going_to_fill()
                ishuffled_address = su.address_generator(self._setting, 'IShuffled', train_mode=self._train_mode, number_of_images_per_chunk=self._number_of_images_per_chunk,
                                                         samples_per_image=self._samples_per_image, semi_epoch=self._semi_epoch, chunk=self._chunk, stage=self._stage)
            if not self._semi_epoch:
                self.fill()
                self.go_to_next_chunk()
        logging.debug('Direct1stEpoch: exiting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')

    def fill(self):
        number_of_images_per_chunk = self._number_of_images_per_chunk
        if self._train_mode == 'Training':
            self._dvf_list = [None] * number_of_images_per_chunk
        if self._semi_epochs_completed:  # This never runs
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
            self._dvf_list = [None] * number_of_images_per_chunk

        torso_list = [None] * len(self._dvf_list)
        indices_chunk = random_indices[lower_range: upper_range]
        im_info_list = [im_info_list_full[i] for i in indices_chunk]
        for i_index_im, index_im in enumerate(indices_chunk):
            _, _, self._dvf_list[i_index_im], torso_list[i_index_im] = \
                synth.get_dvf_and_deformed_images(self._setting,
                                                  im_info=im_info_list_full[index_im],
                                                  stage=self._stage,
                                                  mode_synthetic_dvf='generation'
                                                  )
            if self._setting['verbose']:
                logging.debug('Direct1stEpoch: Data='+im_info_list_full[index_im]['data'] +
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
            # self._ishuffled = 1
            np.save(ishuffled_address, self._ishuffled)

    def go_to_next_chunk(self):
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0
        else:
            self._chunk = self._chunk + 1
        logging.debug('direct1stEpoch: NextChunk, TrainMode=' + self._train_mode + ', semiEpoch={}, Chunk={}, batchCounter={} '.format(self._semi_epoch, self._chunk, self._batch_counter))

    def go_to_next_chunk_without_going_to_fill(self):
        im_info_list_full = copy.deepcopy(self._im_info_list_full)
        upper_range = ((self._chunk+1) * self._number_of_images_per_chunk)
        if upper_range >= len(im_info_list_full):
            self._semi_epochs_completed = 1
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0
        else:
            self._chunk = self._chunk + 1
        logging.debug('Direct1stEpoch: NextChunk, TrainMode = ' + self._train_mode + ' semiEpoch = {}, Chunk = {}, batchCounter = {}  '.format(self._semi_epoch, self._chunk, self._batch_counter))

    def write_log_images_per_chunk(self, pairSelection):
        logFileName = self._IshName.replace('.npy','.txt')
        # with open (logFileName) as logFile:
        #     print("  validation loss:\t\t{:.6f}".format(val_err / val_batches), file=logFile)
        #
        # for i, pair in enumerate(pairSelection):
        #     SyntheticDVFClass = syndef.SyntheticDVF( setting=self._setting,
        #         ImageType=ImageTypeTotal[pair],  # 0: Fixed image, 1: Moving image
        #         IN=INTotal[pair],             # number of the image in the database. In SPREAD database it can be between 1 and 21. (Please note that it starts from 1 not 0)
        #         DeformName=self._setting['deformName'],  # Name of the folder to write or to read.
        #         Dsmooth=DsmoothTotal[pair],              # This variable is used to generate another deformed version of the moving image. Then, use that image to make synthetic DVFs. More information available on [sokooti2017nonrigid]
        #         D=DsmoothTotal[pair]%len(deformMethod),  # 0: low freq, 1: medium freq, 2: high freq. More information available on [sokooti2017nonrigid])
        #         mode='generation'