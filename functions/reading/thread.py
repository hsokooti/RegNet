import copy
import logging
import numpy as np
import os
import threading
import time
import functions.artificial_generation.dvf_generation as synth
from . import utils as reading_utils
import functions.setting_utils as su


class Images(threading.Thread):
    def __init__(self,
                 setting=None,
                 number_of_images_per_chunk=None,  # number of images that I would like to load in RAM
                 samples_per_image=None,
                 im_info_list_full=None,
                 train_mode=None,
                 semi_epoch=0,
                 stage=None,
                 ):
        threading.Thread.__init__(self)
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.daemon = True

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
        self._filled = 0
        if stage is None:
            stage = setting['stage']
        self._stage = stage
        self._ishuffled = None
        ishuffled_folder = su.address_generator(setting, 'IShuffledFolder', train_mode=train_mode, stage=stage)
        if not (os.path.isdir(ishuffled_folder)):
            os.makedirs(ishuffled_folder)

    def run(self):
        # Borrowed from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                self.fill()
            time.sleep(1)

    def pause(self):
        # Modified from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        if not self.paused:
            self.paused = True
            # If in sleep, we acquire immediately, otherwise we wait for thread
            # to release condition. In race, worker will still see self.paused
            # and begin waiting until it's set back to False
            self.pause_cond.acquire()

    def resume(self):
        # Modified from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        if self.paused:
            self.paused = False
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()

    def fill(self):
        self._filled = 0
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
                                                  mode_synthetic_dvf='reading'
                                                  )
            if self._setting['verbose']:
                logging.debug('thread: Data='+im_info_list_full[index_im]['data'] +
                              ', TypeIm={}, CN={}, Dsmooth={}, stage={} is loaded'.format(im_info_list_full[index_im]['type_im'], im_info_list_full[index_im]['cn'],
                                                                                          im_info_list_full[index_im]['dsmooth'], self._stage))
        ishuffled_address = su.address_generator(self._setting, 'IShuffled', train_mode=self._train_mode, number_of_images_per_chunk=self._number_of_images_per_chunk,
                                                 samples_per_image=self._samples_per_image, semi_epoch=self._semi_epoch, chunk=self._chunk, stage=self._stage)
        if self._semi_epoch == 0:
            # in semiEpoch = 0 we wait for the direct_1st_epoch to creat the ishuffled!
            countWait = 1
            while not os.path.isfile(ishuffled_address):
                time.sleep(5)
                logging.debug('thread: waiting {} s for IShuffled:'.format(countWait*5) + ishuffled_address)
                countWait += 1
            self._ishuffled = np.load(ishuffled_address)
        else:
            if os.path.isfile(ishuffled_address):
                self._ishuffled = np.load(ishuffled_address)
            else:
                self._ishuffled = reading_utils.shuffled_indices_from_chunk(self._setting, dvf_list=self._dvf_list, torso_list=torso_list, im_info_list=im_info_list,
                                                                            stage=self._stage, semi_epoch=self._semi_epoch, chunk=self._chunk, samples_per_image=self._samples_per_image,
                                                                            number_of_images_per_chunk=number_of_images_per_chunk, log_header='direct')
                np.save(ishuffled_address, self._ishuffled)

        self._filled = 1
        logging.debug ('Thread is filled .....................')
        self.pause()

    def go_to_next_chunk(self):
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0
        else:
            # imagine that we are in the chunk 16 here
            self._chunk = self._chunk + 1
            # now the chunk is set to 17. assume that this is the last chunk and only one image is available, this can cause some problem
            # INTotal = np.tile(np.repeat(self._INList, len(self._setting['DsmoothList'])), 2)
            # upperRange = ((self._chunk + 1) * self._numberOfImagesPerChunk)
            # lowerRange = ((self._chunk) * self._numberOfImagesPerChunk)
            # if (upperRange - lowerRange) < 5:
            #     self._semiEpoch = self._semiEpoch + 1
            #     self._semiEpochs_completed = 0
            #     self._chunk = 0

        logging.debug('thread: NextChunk, train_mode = ' + self._train_mode + ' semiEpoch = {}, Chunk = {}, batchCounter = {}  '.format(self._semi_epoch, self._chunk, self._batch_counter))

