import numpy as np
import time
import threading
from queue import Queue
import logging
import functions.setting.setting_utils as su
from . import chunk_image_seq


class FillPatches(threading.Thread):
    def __init__(self,
                 setting,
                 batch_size=None,
                 max_queue_size=None,
                 stage_sequence=None,
                 semi_epoch=None,
                 full_image=None,
                 ):
        if batch_size is None:
            batch_size = setting['NetworkTraining']['BatchSize']
        if max_queue_size is None:
            max_queue_size = setting['NetworkTraining']['MaxQueueSize']
        if stage_sequence is None:
            stage_sequence = setting['stage_sequence']
        if full_image is None:
            full_image = setting['FullImage']
        threading.Thread.__init__(self)
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.daemon = True
        self._batch_size = batch_size
        self._stage_sequence = stage_sequence
        self._full_image = full_image
        im_list_info = su.get_im_info_list_from_train_mode(setting, train_mode='Training', load_mode='Single')

        self._chunk_thread = chunk_image_seq.Images(setting=setting,
                                                    class_mode='Thread',
                                                    number_of_images_per_chunk=setting['NetworkTraining']['NumberOfImagesPerChunk'],
                                                    samples_per_image=setting['NetworkTraining']['SamplesPerImage'],
                                                    im_info_list_full=im_list_info,
                                                    stage_sequence=self._stage_sequence,
                                                    semi_epoch=semi_epoch,
                                                    train_mode='Training',
                                                    full_image=full_image)
        self._chunk_thread.start()
        while not self._chunk_thread._filled:
            time.sleep(5)
        self._chunk_direct = chunk_image_seq.Images(setting=setting,
                                                    class_mode='Direct',
                                                    number_of_images_per_chunk=setting['NetworkTraining']['NumberOfImagesPerChunk'],
                                                    samples_per_image=setting['NetworkTraining']['SamplesPerImage'],
                                                    im_info_list_full=im_list_info,
                                                    stage_sequence=self._stage_sequence,
                                                    train_mode='Training',
                                                    full_image=full_image)
        self._chunk_direct.copy_chunk_image(self._chunk_thread)
        self._chunks_completed = False
        self._chunk_thread._filled = 0
        self._thread_is_filling = False
        self._PatchQueue = Queue(maxsize=max_queue_size)

    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    if self._chunk_thread._filled:
                        self._thread_is_filling = False
                    if self._chunks_completed:
                        if not self._chunk_thread._filled:
                            logging.debug('TrainQueue: Training the network is faster than reading the data ..... please wait .....')
                            while (not self._chunk_thread._filled):
                                time.sleep(2)
                        else:
                            logging.debug('TrainQueue: Training the network is slower than reading the data  :-) ')

                        self._chunk_direct = chunk_image_seq.Images(setting=self._chunk_thread._setting,
                                                                    class_mode='Direct',
                                                                    number_of_images_per_chunk=self._chunk_thread._number_of_images_per_chunk,
                                                                    samples_per_image=self._chunk_thread._samples_per_image,
                                                                    im_info_list_full=self._chunk_thread._im_info_list_full,
                                                                    stage_sequence=self._stage_sequence,
                                                                    train_mode=self._chunk_thread._train_mode,
                                                                    full_image=self._chunk_thread._full_image)
                        self._chunk_direct.copy_chunk_image(self._chunk_thread)
                        self._chunk_thread._filled = 0
                        logging.debug('TrainQueue: thread._filled is set to 0')
                        self._chunks_completed = False
                        self._thread_is_filling = False

                    if (not self._chunk_thread._filled) and (not self._thread_is_filling):
                        self._chunk_thread.go_to_next_chunk()
                        self._chunk_thread.resume()
                        self._thread_is_filling = True
                        logging.debug('TrainQueue: thread is resumed')
                    time_before_put = time.time()
                    if self._chunk_thread._setting['Augmentation']:
                        batch_im_orig, batch_dvf_orig = self._chunk_direct.next_batch(self._batch_size)
                        if self._chunk_thread._semi_epoch % 5 == 0:
                            batch_im, batch_dvf = batch_im_orig, batch_dvf_orig
                        elif self._chunk_thread._semi_epoch % 5 == 1:
                            batch_im = np.flip(batch_im_orig.copy(), 1)
                            where_negative = np.ones(batch_dvf_orig.shape, dtype=np.int8)
                            where_negative[:, :, :, :, 0] = -1
                            batch_dvf = batch_dvf_orig * where_negative

                            # np.negative generates some NAN elements. no idea why
                            # where_negative = np.zeros(batch_dvf_orig.shape, dtype=np.bool)
                            # where_negative[:, :, :, :, 0] = True
                            # batch_dvf = np.negative(batch_dvf_orig.copy(), where=where_negative)

                        elif self._chunk_thread._semi_epoch % 5 == 2:
                            batch_im = np.flip(batch_im_orig.copy(), 2)
                            where_negative = np.ones(batch_dvf_orig.shape, dtype=np.int8)
                            where_negative[:, :, :, :, 1] = -1
                            batch_dvf = batch_dvf_orig * where_negative

                        elif self._chunk_thread._semi_epoch % 5 == 3:
                            batch_im = np.flip(batch_im_orig.copy(), 3)
                            where_negative = np.ones(batch_dvf_orig.shape, dtype=np.int8)
                            where_negative[:, :, :, :, 2] = -1
                            batch_dvf = batch_dvf_orig * where_negative

                        elif self._chunk_thread._semi_epoch % 5 == 4:
                            # in np 1.15 you can give all the axes at once: batch_im = np.flip(batch_im_orig, (1, 2, 3))
                            batch_im = np.flip(batch_im_orig.copy(), 1)
                            batch_im = np.flip(batch_im, 2)
                            batch_im = np.flip(batch_im, 3)
                            batch_dvf = -batch_dvf_orig.copy()

                        self._PatchQueue.put((batch_im, batch_dvf))
                    else:
                        batch_both, batch_dvf = self._chunk_direct.next_batch(self._batch_size)
                        if np.shape(batch_both['stage1'])[0] == 15:
                            self._PatchQueue.put((batch_both, batch_dvf))
                            logging.debug('TrainQueue: put patch size:{}'.format(np.shape(batch_both)))

                    time_after_put = time.time()
                    logging.debug('TrainQueue: put {:.2f} s'.format(time_after_put - time_before_put))
                    if self._chunk_direct._chunks_completed:
                        logging.debug('TrainQueue: chunk is completed')
                        self._chunks_completed = True

                    if self._PatchQueue.full():
                        logging.debug('TrainQueue: thread is paused')
                        self.pause()

                finally:
                    time.sleep(0.1)

    def pause(self):
        if not self.paused:
            self.paused = True
            # If in sleep, we acquire immediately, otherwise we wait for thread
            # to release condition. In race, worker will still see self.paused
            # and begin waiting until it's set back to False
            self.pause_cond.acquire()

    def resume(self):
        if self.paused:
            self.paused = False
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
