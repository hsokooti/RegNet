import copy
import logging
import numpy as np
import os
import threading
import time
from . import utils as reading_utils
import functions.artificial_generation as ag
import functions.setting.setting_utils as su


class Images(threading.Thread):
    def __init__(self,
                 class_mode=None,                   # '1stEpoch', 'Thread', 'Direct'
                 setting=None,
                 number_of_images_per_chunk=None,   # number of images that I would like to load in RAM
                 samples_per_image=None,            # number of patches per image
                 im_info_list_full=None,
                 train_mode=None,                   # 'Training' or 'Validation'
                 semi_epoch=0,
                 stage=None,
                 full_image=None,
                 ):
        """
        This class is used to read a chunk of images. It is an inheritance of threading.Thread but actually it is not always a thread.
        It has 3 mode: '1stEpoch': Potentially can be run as a new process. The idea is to generate the images in the first epoch
                                   in parallel and then just kill this process. Please note that the needed memory will be doubled.
                        'Thread' : To be used as a thread.
                        'Direct' : To be used in a normal way. No thread no process

        _mode_artificial_generation:
            train_mode = 'Training'
                '1stEpoch' :                                              'generation'
                'Thread' and setting['ParallelGeneration1stEpoch']=False: 'generation'
                'Thread' and setting['ParallelGeneration1stEpoch']=True:  'reading'
                'Direct':                                                 'reading'
            train_mode = 'Validation'
                'generation'
        :param setting:
        :param number_of_images_per_chunk: number of images that I would like to load in RAM
        :param samples_per_image:
        :param im_info_list_full:
        :param train_mode:
        :param semi_epoch:
        :param stage:
        :param class_mode: '1stEpoch', 'Thread', 'Direct'
        """
        if class_mode == 'Thread':
            threading.Thread.__init__(self)
            self.paused = False
            self.pause_cond = threading.Condition(threading.Lock())
            self.daemon = True
            self._filled = 0

        self._setting = setting
        self._number_of_images_per_chunk = number_of_images_per_chunk
        self._samples_per_image = samples_per_image
        self._chunk = self._setting['start_chunk']
        self._chunks_completed = 0
        self._semi_epochs_completed = 0
        self._semi_epoch = semi_epoch
        self._batch_counter = 0
        self._class_mode = class_mode
        self._fixed_im_list = [None] * number_of_images_per_chunk
        self._deformed_im_list = [None] * number_of_images_per_chunk
        self._dvf_list = [None] * number_of_images_per_chunk
        self._im_info_list_full = im_info_list_full
        self._train_mode = train_mode
        if stage is None:
            stage = setting['stage']
        self._stage = stage
        self._full_image = full_image
        if self._train_mode == 'Training':
            if self._class_mode == '1stEpoch':
                self._mode_artificial_generation = 'generation'
            elif self._class_mode == 'Thread'and not self._setting['ParallelGeneration1stEpoch']:
                if not self._setting['never_generate_image']:
                    self._mode_artificial_generation = 'generation'
                else:
                    self._mode_artificial_generation = 'reading'

            elif self._class_mode == 'Thread'and self._setting['ParallelGeneration1stEpoch']:
                self._mode_artificial_generation = 'reading'

            elif self._class_mode == 'Direct':
                self._mode_artificial_generation = 'reading'

        elif self._train_mode == 'Validation':
            self._mode_artificial_generation = 'generation'
        self._ishuffled = None

    def run(self):
        """ only in Thread mode
        Borrowed from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        """
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                self.fill()
            time.sleep(1)

    def pause(self):
        """ only in Thread mode """
        if not self.paused:
            self.paused = True
            # If in sleep, we acquire immediately, otherwise we wait for thread
            # to release condition. In race, worker will still see self.paused
            # and begin waiting until it's set back to False
            self.pause_cond.acquire()

    def resume(self):
        """ only in Thread mode """
        if self.paused:
            self.paused = False
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()

    def generate_chunk_only(self):
        # only in 1stEpoch mode
        while self._semi_epoch == 0:
            ishuffled_folder = reading_utils.get_ishuffled_folder_write_ishuffled_setting(self._setting, self._train_mode, self._stage,
                                                                                          self._number_of_images_per_chunk,
                                                                                          self._samples_per_image, self._im_info_list_full,
                                                                                          full_image=self._full_image)
            ishuffled_name = su.address_generator(self._setting, 'IShuffledName', semi_epoch=self._semi_epoch, chunk=self._chunk)
            ishuffled_address = ishuffled_folder + ishuffled_name

            while os.path.isfile(ishuffled_address) and not self._semi_epoch:
                logging.debug(self._class_mode+': stage={}, SemiEpoch={}, Chunk={} is already generated, going to next chunk'.format(self._stage, self._semi_epoch, self._chunk))
                self.go_to_next_chunk_without_going_to_fill()
                ishuffled_name = su.address_generator(self._setting, 'IShuffledName', semi_epoch=self._semi_epoch, chunk=self._chunk)
                ishuffled_address = ishuffled_folder + ishuffled_name

            if not self._semi_epoch:
                self.fill()
                self.go_to_next_chunk()
        logging.debug(self._class_mode+': generate_chunk_only exiting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')

    def generate_image_only(self):
        """
        check if all images in self._im_info_list_full are available or generate them. Does not search for indices and create ishuffled.
        :return:
        """
        im_info_list_full = copy.deepcopy(self._im_info_list_full)
        random_state = np.random.RandomState(self._semi_epoch)
        if self._setting['Randomness']:
            random_indices = random_state.permutation(len(im_info_list_full))
        else:
            random_indices = np.arange(len(im_info_list_full))

        im_info_list = [im_info_list_full[i] for i in random_indices]

        for i_im_info, im_info in enumerate(im_info_list):
            mask_to_zero = self._setting['deform_exp'][im_info['deform_exp']]['MaskToZero']
            if not ag.check_all_images_exist(self._setting, im_info, self._stage, mask_to_zero=mask_to_zero):
                ag.get_dvf_and_deformed_images(self._setting,
                                               im_info=im_info,
                                               stage=self._stage,
                                               mode_synthetic_dvf=self._mode_artificial_generation
                                               )
            logging.debug(self._class_mode+', image {}/{} is done'.format(i_im_info, len(im_info_list)))
        logging.debug(self._class_mode+': generate_image_only exiting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')

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
        random_state = np.random.RandomState(self._semi_epoch)
        if self._setting['Randomness']:
            random_indices = random_state.permutation(len(im_info_list_full))
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

        log_msg = self._class_mode+': stage={}, SemiEpoch={}, Chunk={} '.format(self._stage, self._semi_epoch, self._chunk)
        logging.debug(log_msg)
        if self._class_mode == 'Thread':
            with open(su.address_generator(self._setting, 'log_im_file'), 'a+') as f:
                f.write(log_msg + '\n')

        torso_list = [None] * len(self._dvf_list)
        indices_chunk = random_indices[lower_range: upper_range]
        im_info_list = [im_info_list_full[i] for i in indices_chunk]
        for i_index_im, index_im in enumerate(indices_chunk):
            self._fixed_im_list[i_index_im], self._deformed_im_list[i_index_im], self._dvf_list[i_index_im], torso_list[i_index_im] = \
                ag.get_dvf_and_deformed_images(self._setting,
                                               im_info=im_info_list_full[index_im],
                                               stage=self._stage,
                                               mode_synthetic_dvf=self._mode_artificial_generation
                                               )
            if self._class_mode == '1stEpoch':
                self._fixed_im_list[i_index_im] = None
                self._deformed_im_list[i_index_im] = None
            if self._setting['verbose']:
                log_msg = self._class_mode+': Data='+im_info_list_full[index_im]['data'] +\
                          ', TypeIm={}, CN={}, Dsmooth={}, stage={} is loaded'.format(im_info_list_full[index_im]['type_im'], im_info_list_full[index_im]['cn'],
                                                                                      im_info_list_full[index_im]['dsmooth'], self._stage)
                logging.debug(log_msg)
                if self._class_mode == 'Thread':
                    with open(su.address_generator(self._setting, 'log_im_file'), 'a+') as f:
                        f.write(log_msg + '\n')

        ishuffled_folder = reading_utils.get_ishuffled_folder_write_ishuffled_setting(self._setting, self._train_mode, self._stage,
                                                                                      self._number_of_images_per_chunk, self._samples_per_image,
                                                                                      self._im_info_list_full, full_image=self._full_image)
        ishuffled_name = su.address_generator(self._setting, 'IShuffledName', semi_epoch=self._semi_epoch, chunk=self._chunk)
        ishuffled_address = ishuffled_folder + ishuffled_name

        if self._mode_artificial_generation == 'reading' and not (self._class_mode == 'Thread' and self._semi_epoch > 0) and not self._setting['never_generate_image']:
            count_wait = 1
            while not os.path.isfile(ishuffled_address):
                time.sleep(5)
                logging.debug(self._class_mode+': waiting {} s for IShuffled:'.format(count_wait*5) + ishuffled_address)
                count_wait += 1
            self._ishuffled = np.load(ishuffled_address)

        if os.path.isfile(ishuffled_address):
            self._ishuffled = np.load(ishuffled_address)
            log_msg = self._class_mode + ': loading IShuffled: ' + ishuffled_address
        else:
            log_msg = self._class_mode + ': generating IShuffled: ' + ishuffled_address
            self._ishuffled = reading_utils.shuffled_indices_from_chunk(self._setting, dvf_list=self._dvf_list, torso_list=torso_list,
                                                                        im_info_list=im_info_list, stage=self._stage, semi_epoch=self._semi_epoch,
                                                                        chunk=self._chunk, samples_per_image=self._samples_per_image,
                                                                        log_header=self._class_mode, full_image=self._full_image)
            np.save(ishuffled_address, self._ishuffled)
            logging.debug(self._class_mode + ': saving IShuffled: ' + ishuffled_address)

        logging.debug(log_msg)
        if self._class_mode == 'Thread':
            with open(su.address_generator(self._setting, 'log_im_file'), 'a+') as f:
                f.write(log_msg + '\n')
        if self._class_mode == 'Thread':
            if not self._full_image:
                class_balanced = self._setting['ClassBalanced']
                hist_class = np.zeros(len(class_balanced), dtype=np.int32)
                hist_text = ''
                for c in range(len(class_balanced)):
                    hist_class[c] = sum(self._ishuffled[:, 2] == c)
                    hist_text = hist_text + 'Class'+str(c)+': '+str(hist_class[c]) + ', '
                log_msg = hist_text+self._class_mode+': stage={}, SemiEpoch={}, Chunk={} '.format(self._stage, self._semi_epoch, self._chunk)
                with open(su.address_generator(self._setting, 'log_im_file'), 'a+') as f:
                    f.write(log_msg + '\n')

            with open(su.address_generator(self._setting, 'log_im_file'), 'a+') as f:
                f.write('========================' + '\n')
            logging.debug('Thread is filled .....will be paused')
            self._filled = 1
            self.pause()

    def go_to_next_chunk(self):
        if self._semi_epochs_completed:
            self._semi_epoch = self._semi_epoch + 1
            self._semi_epochs_completed = 0
            self._chunk = 0
        else:
            self._chunk = self._chunk + 1
        logging.debug(self._class_mode+': NextChunk, TrainMode=' + self._train_mode + ', SemiEpoch={}, Chunk={}, batchCounter={} '.
                      format(self._semi_epoch, self._chunk, self._batch_counter))

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
        logging.debug(self._class_mode+': NextChunk, TrainMode = ' + self._train_mode + ' SemiEpoch = {}, Chunk = {}, batchCounter = {}  '.
                      format(self._semi_epoch, self._chunk, self._batch_counter))

    def next_batch(self, batch_size):
        if self._chunks_completed:
            self._chunk = self._chunk + 1
            self.fill()
            self._batch_counter = 0
            self._chunks_completed = 0
        end_batch = (self._batch_counter + 1) * batch_size
        if end_batch >= len(self._ishuffled):
            self._chunks_completed = 1
            end_batch = len(self._ishuffled)
        batch_both, batch_dvf = reading_utils.extract_batch(self._setting, stage=self._stage,
                                                            ish=self._ishuffled,
                                                            fixed_im_list=self._fixed_im_list,
                                                            deformed_im_list=self._deformed_im_list,
                                                            dvf_list=self._dvf_list,
                                                            batch_counter=self._batch_counter,
                                                            batch_size=batch_size,
                                                            end_batch=end_batch,
                                                            full_image=self._full_image)
        if self._setting['verbose']:
            logging.debug(self._class_mode+': TrainMode='+self._train_mode+', SemiEpoch={}, Chunk={}, BatchCounter={} , EndBatch={} '.
                          format(self._semi_epoch, self._chunk, self._batch_counter, end_batch))
        self._batch_counter = self._batch_counter + 1
        return batch_both, batch_dvf

    def reset_validation(self):
        self._batch_counter = 0
        self._chunks_completed = 0
        self._chunk = 0
        self._semi_epochs_completed = 0
        self._semi_epoch = 0

    def copy_chunk_image(self, ref_object):
        self._setting = copy.deepcopy(ref_object._setting)
        self._number_of_images_per_chunk = copy.copy(ref_object._number_of_images_per_chunk)
        self._samples_per_image = copy.copy(ref_object._samples_per_image)
        self._batch_counter = 0
        self._chunk = copy.copy(ref_object._chunk)
        self._chunks_completed = copy.copy(ref_object._chunks_completed)
        self._semi_epochs_completed = copy.copy(ref_object._semi_epochs_completed)
        self._semi_epoch = copy.copy(ref_object._semi_epoch)
        self._fixed_im_list = copy.deepcopy(ref_object._fixed_im_list)
        self._deformed_im_list = copy.deepcopy(ref_object._deformed_im_list)
        self._dvf_list = copy.deepcopy(ref_object._dvf_list)
        self._im_info_list_full = copy.deepcopy(ref_object._im_info_list_full)
        self._train_mode = copy.copy(ref_object._train_mode)
        self._ishuffled = copy.deepcopy(ref_object._ishuffled)
        self._stage = copy.copy(ref_object._stage)
        self._full_image = copy.copy(ref_object._full_image)
