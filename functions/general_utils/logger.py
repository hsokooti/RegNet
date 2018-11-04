import logging
import os
import sys
import time


class Logger(object):
    # Borrowed from: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    def __init__(self, log_address=''):
        self.terminal = sys.stdout
        self.log = open(log_address, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    modified from: https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
    """
    def __init__(self, logger, log_level=logging.DEBUG):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def set_log_file(log_file_name, short_mode=False):
    log_folder = log_file_name.rsplit('/', 1)[0]
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    if short_mode:
        logging.basicConfig(
            format='[%(message)s',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(log_file_name),
                logging.StreamHandler()
            ])
    else:
        logging.basicConfig(
            format='[%(threadName)-12.12s] %(message)s',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(log_file_name),
                logging.StreamHandler()
            ])

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.DEBUG)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.DEBUG)
    sys.stderr = sl

    logging.debug('---------------------------------{}--------------------------------'.format(time.ctime()))
    logging.debug('----------------------------------start experiment------------------------------')
