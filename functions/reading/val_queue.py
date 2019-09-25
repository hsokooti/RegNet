import copy
import time
import threading
from queue import Queue
from . import direct
import logging
import functions.setting.setting_utils as su


class FillPatches(threading.Thread):
    def __init__(self,
                 setting,
                 stage=None,
                 batch_size=None,
                 max_queue_size=None
                 ):
        if batch_size is None:
            batch_size = setting['NetworkValidation']['BatchSize']
        if max_queue_size is None:
            max_queue_size = setting['NetworkValidation']['MaxQueueSize']
        if stage is None:
            stage = setting['stage']
        threading.Thread.__init__(self)
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.daemon = True
        self._batchSize = batch_size
        self._chunks_completed = False
        self._PatchQueue = Queue(maxsize=max_queue_size)
        im_list_info = su.get_im_info_list_from_train_mode(setting, train_mode='Validation')
        self._reading = direct.Images(setting=setting,
                                      number_of_images_per_chunk=setting['NetworkValidation']['NumberOfImagesPerChunk'],
                                      samples_per_image=setting['NetworkValidation']['SamplesPerImage'],
                                      im_info_list_full=im_list_info,
                                      stage=stage,
                                      train_mode='Validation')
        self._reading.fill()

    def run(self):
        # Borrowed from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    time_before_put = time.time()
                    item_queue = self._reading.next_batch(self._batchSize) + (copy.copy(self._reading._chunks_completed),)
                    self._PatchQueue.put(item_queue)
                    time_after_put = time.time()
                    logging.debug('valQueue: put {:.2f} s'.format(time_after_put - time_before_put))
                    if self._reading._chunks_completed:
                        logging.debug('valQueue: chunk is completed: resetValidation() ')
                        self._chunks_completed = True
                        self._reading.reset_validation()
                    if self._PatchQueue.full():
                        self.pause()
                finally:
                    time.sleep(0.1)

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
