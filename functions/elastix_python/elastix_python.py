import os
import numpy as np


def transformix(parameter_file, output_directory, points=None, transformix_address='transformix', input_image=None,
                threads=None, run_mode=True):
    trx_cmd = transformix_address + ' -tp ' + parameter_file + ' -out ' + output_directory
    if points is not None:
        trx_cmd = trx_cmd + ' -def ' + points
    if input_image is not None:
        trx_cmd = trx_cmd + ' -in ' + input_image
    if threads is not None:
        trx_cmd = trx_cmd + ' -threads ' + str(threads)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if run_mode:
        os.system(trx_cmd)
    return trx_cmd


def elastix(parameter_file, output_directory, fixed_image, moving_image, elastix_address='elastix',
            fixed_mask=None, moving_mask=None, initial_transform=None, threads=None, run_mode=True):
    elx_cmd = elastix_address + ' -f ' + fixed_image + ' -m ' + moving_image + ' -out ' + output_directory + \
              ' -p ' + parameter_file
    if initial_transform is not None:
        elx_cmd = elx_cmd + ' -t0 ' + initial_transform
    if fixed_mask is not None:
        elx_cmd = elx_cmd + ' -fMask ' + fixed_mask
    if moving_mask is not None:
        elx_cmd = elx_cmd + ' -mMask ' + moving_mask
    if threads is not None:
        elx_cmd = elx_cmd + ' -threads ' + str(threads)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if run_mode:
        os.system(elx_cmd)
    return elx_cmd


class elxReadOutputPointsFile(object):
    def __init__(self, pointPath):
        self.rawFile = np.genfromtxt(pointPath, delimiter=';', dtype='str')
        self.Index = np.zeros(np.shape(self.rawFile)[0], dtype=np.int16)
        self.InputIndex = np.zeros((np.shape(self.rawFile)[0], 3), dtype=np.int16)
        self.InputPoint = np.zeros((np.shape(self.rawFile)[0], 3), dtype=np.float32)
        self.OutputIndexFixed = np.zeros((np.shape(self.rawFile)[0], 3), dtype=np.int16)
        self.OutputPoint = np.zeros((np.shape(self.rawFile)[0], 3), dtype=np.float32)
        self.Deformation = np.zeros((np.shape(self.rawFile)[0], 3), dtype=np.float32)
        self.splitPointsFile()

    def splitPointsFile(self):
        rawFile = self.rawFile
        for i in range(np.shape(rawFile)[0]):
            for j in range(np.shape(rawFile)[1]):
                if rawFile[i, j][0:len('Point')] == 'Point':
                    self.Index[i] = int(rawFile[i, j][len('Point'):])
                if (rawFile[i, j][0:len(' InputIndex')]).strip() == 'InputIndex':
                    splitCell = (rawFile[i, j]).split()
                    self.InputIndex[i,:] = np.array([splitCell[3], splitCell[4], splitCell[5]])
                if (rawFile[i, j][0:len(' InputPoint')]).strip() == 'InputPoint':
                    splitCell = (rawFile[i, j]).split()
                    self.InputPoint[i, :] = np.array([splitCell[3], splitCell[4], splitCell[5]])
                if (rawFile[i, j][0:len(' OutputIndexFixed')]).strip() == 'OutputIndexFixed':
                    splitCell = (rawFile[i, j]).split()
                    self.OutputIndexFixed[i,:] = np.array([splitCell[3], splitCell[4], splitCell[5]])
                if (rawFile[i, j][0:len(' OutputPoint')]).strip() == 'OutputPoint':
                    splitCell = (rawFile[i, j]).split()
                    self.OutputPoint[i, :] = np.array([splitCell[3], splitCell[4], splitCell[5]])
                if (rawFile[i, j][0:len(' Deformation')]).strip() == 'Deformation':
                    splitCell = (rawFile[i, j]).split()
                    self.Deformation[i, :] = np.array([splitCell[3], splitCell[4], splitCell[5]])
