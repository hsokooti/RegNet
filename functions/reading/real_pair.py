import copy
import logging
import numpy as np
import SimpleITK as sitk
import functions.elastix_python as elxpy
import functions.setting.setting_utils as su


class Images(object):
    def __init__(self, setting, pair_info, stage, fixed_im_sitk=None, moved_im_affine_sitk=None, fixed_torso_sitk=None, moved_torso_affine_sitk=None):

        self._setting = setting
        self._pair_info = pair_info
        self._stage = stage

        # landmarks local member
        self._prepared_landmarks_done = 0
        self._chunks_completed = 0
        self._batch_counter = 0
        self._fixed_landmarks_world = None
        self._fixed_landmarks_index = None
        self._fixed_landmarks_index_padded = None           # Sometimes we need to pad the image in order to extract patches
        self._moving_landmarks_world = None                 # moving image: for instance the follow-up image
        self._fixed_after_affine_landmarks_world = None     # self._FixedAfterAffineLandmarksWorld = self._FixedLandmarksWorld + self._DVFAffine
        self._dvf_affine = None
        self._start_batch = None
        self._end_batch = None

        # sweeping local member
        self._padding_done = 0
        self._offset_after_padding = np.array([0, 0, 0])
        self._extra_pad_before = np.array([0, 0, 0])
        self._extra_pad_after = np.array([0, 0, 0])
        self._uneven_padding = np.array([0, 0, 0])
        self._sweep_completed = 0
        self._win_center = np.array([0, 0, 0])
        self._sweep0_completed = 0
        self._sweep1_completed = 0
        self._sweep2_completed = 0

        # loading and preprocess images
        if fixed_im_sitk is None and moved_im_affine_sitk is None:
            self._fixed_im_sitk, self._moved_im_affine_sitk, fixed_torso_sitk, moved_torso_affine_sitk = self.load_pair()
        else:
            self._fixed_im_sitk = fixed_im_sitk
            self._moved_im_affine_sitk = moved_im_affine_sitk
        self._fixed_im = sitk.GetArrayFromImage(self._fixed_im_sitk)
        self._moved_im_affine = sitk.GetArrayFromImage(self._moved_im_affine_sitk)
        self._fixed_im_original_size = np.shape(self._fixed_im)
        if self._setting['torsoMask']:
            self._fixed_torso_sitk = fixed_torso_sitk
            self._fixed_torso = sitk.GetArrayFromImage(fixed_torso_sitk)
            self._moved_torso_affine_sitk = moved_torso_affine_sitk
            self._moved_torso_affine = sitk.GetArrayFromImage(moved_torso_affine_sitk)

        self.preprocess_images()

    def prepare_for_landmarks(self, padding=True):
        im_info_fixed = copy.deepcopy(self._pair_info[0])
        im_info_moving = copy.deepcopy(self._pair_info[1])
        fixed_landmarks_world = np.loadtxt(su.address_generator(self._setting, 'LandmarkPoint_tr',
                                                                pair_info=self._pair_info, **im_info_fixed))
        moving_landmarks_world = np.loadtxt(su.address_generator(self._setting, 'LandmarkPoint_tr',
                                                                 pair_info=self._pair_info, **im_info_moving))
        if self._setting['data'][self._pair_info[0]['data']]['UnsureLandmarkAvailable']:
            fixed_landmarks_unsure_list = np.loadtxt(su.address_generator(self._setting, 'UnsurePoints', **im_info_fixed))
            index_sure = []
            for i in range(len(fixed_landmarks_unsure_list)):
                if fixed_landmarks_unsure_list[i] == 0:
                    index_sure.append(i)
        else:
            index_sure = [i for i in range(len(fixed_landmarks_world))]
        self._fixed_landmarks_world = fixed_landmarks_world[index_sure]      # xyz order
        self._moving_landmarks_world = moving_landmarks_world[index_sure]    # xyz order
        self._fixed_landmarks_index = (np.round((self._fixed_landmarks_world - self._fixed_im_sitk.GetOrigin()) /
                                                np.array(self._fixed_im_sitk.GetSpacing()))).astype(np.int16)  # xyz order

        if self._setting['data'][self._pair_info[1]['data']]['AffineRegistration']:
            elx_all_points = elxpy.elxReadOutputPointsFile(su.address_generator(self._setting, 'reg_AffineOutputPoints',
                                                                                pair_info=self._pair_info, **im_info_fixed))
            self._fixed_after_affine_landmarks_world = elx_all_points.OutputPoint[index_sure]
            self._dvf_affine = elx_all_points.Deformation[index_sure]
        else:
            self._fixed_after_affine_landmarks_world = self._fixed_landmarks_world.copy()
            self._dvf_affine = [[0, 0, 0] for _ in range(len(index_sure))]

        # np.array(self._fixed_im_sitk.GetSpacing())          #xyz order
        # self._fixed_im_sitk.GetOrigin()                     #xyz order

        # The following lines are only used to check the dimension order.
        check_dimension_order = False
        if check_dimension_order:
            dilated_landmarks = np.zeros(np.shape(self._fixed_im), dtype=np.int8)
            rd = 7  # radius of dilation
            for i in range(np.shape(self._fixed_landmarks_index)[0]):
                dilated_landmarks[self._fixed_landmarks_index[i, 2] - rd: self._fixed_landmarks_index[i, 2] + rd,
                                  self._fixed_landmarks_index[i, 1] - rd: self._fixed_landmarks_index[i, 1] + rd,
                                  self._fixed_landmarks_index[i, 0] - rd: self._fixed_landmarks_index[i, 0] + rd] = 1
            dilated_landmarks_sitk = sitk.GetImageFromArray(dilated_landmarks)
            dilated_landmarks_sitk.SetOrigin(self._fixed_im_sitk.GetOrigin())
            dilated_landmarks_sitk.SetSpacing(np.array(self._fixed_im_sitk.GetSpacing()))
            sitk.WriteImage(sitk.Cast(dilated_landmarks_sitk, sitk.sitkInt8),
                            su.address_generator(self._setting, 'DilatedLandmarksIm', **im_info_fixed))

        if padding:
            min_coordinate_landmark = np.min(self._fixed_landmarks_index, axis=0)
            max_coordinate_landmark = np.max(self._fixed_landmarks_index, axis=0)

            pad_before = np.zeros(3, dtype=np.int16)     # xyz order
            pad_after = np.zeros(3, dtype=np.int16)      # xyz order
            for i in range(0, 3):
                # be careful about the xyz or zyx order!
                if min_coordinate_landmark[i] < self._setting['R']:
                    pad_before[i] = self._setting['R'] - min_coordinate_landmark[i]
                if ((np.shape(self._fixed_im)[2-i] - max_coordinate_landmark[i]) - 1) < self._setting['R']:
                    pad_after[i] = self._setting['R'] - (np.shape(self._fixed_im)[2-i] - max_coordinate_landmark[i]) + 1

            self._fixed_im = np.pad(self._fixed_im, ((pad_before[2], pad_after[2]), (pad_before[1], pad_after[1]), (pad_before[0], pad_after[0])),
                                    'constant', constant_values=(self._setting['defaultPixelValue'],))
            self._moved_im_affine = np.pad(self._moved_im_affine, ((pad_before[2], pad_after[2]), (pad_before[1], pad_after[1]), (pad_before[0], pad_after[0])),
                                           'constant', constant_values=(self._setting['defaultPixelValue'],))
            self._fixed_landmarks_index_padded = self._fixed_landmarks_index + pad_before

    def pad_for_sweeping(self):
        pad_before = (self._setting['R']-self._setting['Ry']+self._setting['Ry_erode'])*np.ones(3, dtype=np.int16)     # zyx order
        pad_after = (self._setting['R']-self._setting['Ry']+self._setting['Ry_erode'])*np.ones(3, dtype=np.int16)      # zyx order
        size_after_padding = np.array(np.shape(self._fixed_im)) + pad_before + pad_after
        if self._setting['R'] < np.min(size_after_padding):
            for dim in range(0, 3):
                extra_pad = (size_after_padding[dim] - 2 * self._setting['R'] - 1)
                if extra_pad < 0:
                    if extra_pad % 2 == 1:
                        self._uneven_padding[dim] = 1
                    extra_pad_radius = (-extra_pad) // 2
                    # if the size of the image is too small and is even, we like to make it odd.
                    # We remember that radius on the right side is bigger. so at the test time we know how to crop.
                    self._extra_pad_before[dim] = extra_pad_radius
                    if self._uneven_padding[dim]:
                        self._extra_pad_after[dim] = extra_pad_radius + 1
                        pad_before[dim] = pad_before[dim] + self._extra_pad_before[dim]
                        pad_after[dim] = pad_after[dim] + self._extra_pad_after[dim]
                        logging.debug('Xtra unevenPadding is done. padBefore[{}] = {}, padAfter[{}] = {} '.format(
                            dim, self._extra_pad_before[dim], dim, self._extra_pad_after[dim]))
                    else:
                        self._extra_pad_after[dim] = extra_pad_radius
                        pad_before[dim] = pad_before[dim] + self._extra_pad_before[dim]
                        pad_after[dim] = pad_after[dim] + self._extra_pad_after[dim]
                        logging.debug('Xtra evenPadding is done. padBefore[{}] = {}, padAfter[{}] = {} '.format(
                            dim, self._extra_pad_before[dim], dim, self._extra_pad_after[dim]))
                if extra_pad <= 0:  # we should consider both negative and zero extra_pad. which means that we cannot sweep in that direction anymore.
                    if dim == 0:
                        self._sweep0_completed = 1
                    if dim == 1:
                        self._sweep1_completed = 1
                    if dim == 2:
                        self._sweep2_completed = 1
        self._offset_after_padding = pad_before
        self._fixed_im = np.pad(self._fixed_im, ((pad_before[0], pad_after[0]),
                                                 (pad_before[1], pad_after[1]),
                                                 (pad_before[2], pad_after[2])),
                                'constant',
                                constant_values=(self._setting['data'][self._pair_info[0]['data']]['defaultPixelValue'],))
        self._moved_im_affine = np.pad(self._moved_im_affine, ((pad_before[0], pad_after[0]),
                                                               (pad_before[1], pad_after[1]),
                                                               (pad_before[2], pad_after[2])),
                                       'constant',
                                       constant_values=(self._setting['data'][self._pair_info[1]['data']]['defaultPixelValue'],))

    def preprocess_images(self):
        if self._setting['normalization'] == 'linear':
            self._fixed_im = ((self._fixed_im + 1000) / 4095.).astype(np.float32)
            self._moved_im_affine = ((self._moved_im_affine + 1000) / 4095.).astype(np.float32)
        if self._setting['torsoMask']:
            self._fixed_im[self._fixed_torso == 0] = self._setting['data'][self._pair_info[0]['data']]['defaultPixelValue']
            self._moved_im_affine[self._moved_torso_affine == 0] = self._setting['data'][self._pair_info[0]['data']]['defaultPixelValue']

    def load_pair(self):
        im_info_fixed = copy.deepcopy(self._pair_info[0])
        im_info_fixed['stage'] = self._stage
        im_info_moving = copy.deepcopy(self._pair_info[1])
        im_info_moving['stage'] = self._stage
        fixed_torso_sitk = None
        moved_torso_affine_sitk = None

        if 'dsmooth' in im_info_moving:
            # in this case it means that images are synthetic
            fixed_im_sitk = sitk.ReadImage(su.address_generator(self._setting, 'Im', **im_info_fixed))
            moved_im_affine_sitk = sitk.ReadImage(su.address_generator(self._setting, 'deformedIm', **im_info_moving))
            if self._setting['torsoMask']:
                fixed_torso_sitk = sitk.ReadImage(su.address_generator(self._setting, 'Torso', **im_info_fixed))
                moved_torso_affine_sitk = sitk.ReadImage(su.address_generator(self._setting, 'deformedTorso', **im_info_moving))
        else:
            fixed_im_sitk = sitk.ReadImage(su.address_generator(self._setting, 'originalIm', **im_info_fixed))
            moved_im_affine_sitk = sitk.ReadImage(su.address_generator(self._setting, 'MovedImAffine', pair_info=self._pair_info, **im_info_moving))
            if self._setting['torsoMask']:
                fixed_torso_sitk = sitk.ReadImage(su.address_generator(self._setting, 'originalTorso', **im_info_fixed))
                moved_torso_affine_sitk = sitk.ReadImage(su.address_generator(self._setting, 'MovedTorsoAffine', pair_info=self._pair_info, **im_info_moving))

        return fixed_im_sitk, moved_im_affine_sitk, fixed_torso_sitk, moved_torso_affine_sitk

    def next_landmark_patch(self, batch_size):
        raise RuntimeError('Deprecated')
        # if not self._chunks_completed:
        #     r = self._setting['R']
        #     ry = self._setting['Ry']
        #     endBatch = (self._batchCounter + 1) * batch_size
        #     if endBatch >= len(self._fixed_landmarks_index_padded):
        #         self._chunks_completed = 1
        #         endBatch = len(self._fixed_landmarks_index_padded)
        #     self._start_batch = self._batchCounter * batch_size
        #     self._end_batch = endBatch
        #     if self._setting['Dim'] == '3D':
        #         batchXlow = 0
        #         BatchXFixed = np.stack([self._fixed_im[
        #                                 self._fixed_landmarks_index_padded[i, 2] - r: self._fixed_landmarks_index_padded[i, 2] + r + 1,
        #                                 self._fixed_landmarks_index_padded[i, 1] - r: self._fixed_landmarks_index_padded[i, 1] + r + 1,
        #                                 self._fixed_landmarks_index_padded[i, 0] - r: self._fixed_landmarks_index_padded[i, 0] + r + 1,
        #                                 np.newaxis] for i in range(self._batchCounter * batch_size, endBatch)])
        #         BatchXDMovedImAffine = np.stack([self._moved_im_affine[
        #                                          self._fixed_landmarks_index_padded[i, 2] - r: self._fixed_landmarks_index_padded[i, 2] + r + 1,
        #                                          self._fixed_landmarks_index_padded[i, 1] - r: self._fixed_landmarks_index_padded[i, 1] + r + 1,
        #                                          self._fixed_landmarks_index_padded[i, 0] - r: self._fixed_landmarks_index_padded[i, 0] + r + 1,
        #                                 np.newaxis] for i in range(self._batchCounter * batch_size, endBatch)])
        #
        #         batchX = np.concatenate((BatchXDMovedImAffine, BatchXFixed), axis=4)
        #         batchY = (np.stack([self._moving_landmarks_world[i, :] - self._FixedAfterAffineLandmarksWorld[i, :] for i in range(self._batchCounter * batch_size, endBatch)]))
        #         batchY = np.expand_dims(np.expand_dims(np.expand_dims(batchY, axis=1), 1), 1)
        #     if self._setting['verbose']:
        #         logging.debug('LandmarkPatch: batchCounter = {} , endBatch = {} '.format(self._batchCounter, endBatch))
        #     self._batchCounter = self._batchCounter + 1
        # return batchX, batchY, batchXlow

    def next_sweep_patch(self):
        if not self._padding_done:
            self.pad_for_sweeping()
            self._padding_done = 1

        r = self._setting['R']
        ry = self._setting['Ry']
        ry_erode = self._setting['Ry_erode']
        # if not self._sweep_completed:
        if np.all(self._win_center == [0, 0, 0]):
            self._win_center = [r, r, r]
        if self._setting['Dim'] == '3D':
            batch_fixed = self._fixed_im[np.newaxis,
                                         self._win_center[0] - r: self._win_center[0] + r + 1,
                                         self._win_center[1] - r: self._win_center[1] + r + 1,
                                         self._win_center[2] - r: self._win_center[2] + r + 1,
                                         np.newaxis]
            batch_moved_affine = self._moved_im_affine[np.newaxis,
                                                       self._win_center[0] - r: self._win_center[0] + r + 1,
                                                       self._win_center[1] - r: self._win_center[1] + r + 1,
                                                       self._win_center[2] - r: self._win_center[2] + r + 1,
                                                       np.newaxis]
        else:
            raise RuntimeError('not implemented')
        win_center_without_padding = self._win_center - self._offset_after_padding
        win_r_before = (ry - ry_erode) * np.ones(3, dtype=np.int16) - self._extra_pad_before
        win_r_after = (ry - ry_erode) * np.ones(3, dtype=np.int16) + 1 - self._extra_pad_after
        predicted_begin = ry_erode * np.ones(3, dtype=np.int16) + self._extra_pad_before
        predicted_end = -ry_erode * np.ones(3, dtype=np.int16) - self._extra_pad_after

        if self._setting['verbose']:
            logging.debug('Data=' + self._pair_info[0]['data'] + 'CN = {} win_center = {} '.format(
                self._pair_info[0]['cn'], self._win_center))

        if self._sweep0_completed & self._sweep1_completed:
            self._win_center[0] = r
            self._win_center[1] = r
            self._win_center[2] = self._win_center[2] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[2]+r+1 > np.shape(self._fixed_im)[2]:
                self._win_center[2] = np.shape(self._fixed_im)[2] - r - 1
                if not self._sweep2_completed:
                    self._sweep0_completed = 0
                    self._sweep1_completed = 0
                    self._sweep2_completed = 1
                else:
                    self._sweep_completed = 1
            else:
                self._sweep0_completed = 0
                self._sweep1_completed = 0
        elif self._sweep0_completed:
            self._win_center[0] = r
            self._win_center[1] = self._win_center[1] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[1]+r+1 > np.shape(self._fixed_im)[1]:
                self._win_center[1] = np.shape(self._fixed_im)[1] - r - 1
                self._sweep1_completed = 1
            self._sweep0_completed = 0
        else:
            self._win_center[0] = self._win_center[0] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[0]+r+1 > np.shape(self._fixed_im)[0]:
                self._win_center[0] = np.shape(self._fixed_im)[0] - r - 1
                self._sweep0_completed = 1

        batch_both = np.concatenate((batch_moved_affine, batch_fixed), axis=4)
        if self._setting['verbose']:
            logging.debug('Data=' + self._pair_info[0]['data'] + 'CN = {} next_sweep_patch: win_center = {} '.format(
                self._pair_info[0]['cn'], self._win_center))

        return batch_both, win_center_without_padding, win_r_before, win_r_after, predicted_begin, predicted_end

    def get_fixed_im_sitk(self):
        return self._fixed_im_sitk

    def get_fixed_im(self):
        return self._fixed_im

    def get_moved_im_affine_sitk(self):
        return self._moved_im_affine_sitk

    def get_moved_im_affine(self):
        return self._moved_im_affine

    def get_fixed_torso_sitk(self):
        return self._fixed_torso_sitk

    def get_moved_torso_affine_sitk(self):
        return self._moved_torso_affine_sitk

    def get_sweep_completed(self):
        return self._sweep_completed
