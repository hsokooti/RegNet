import copy
import logging
import numpy as np
import SimpleITK as sitk
import functions.elastix_python as elxpy
import functions.setting.setting_utils as su


class Images(object):
    def __init__(self, setting, pair_info, stage, fixed_im_sitk=None, moved_im_affine_sitk=None, fixed_mask_sitk=None,
                 moved_mask_affine_sitk=None, padto=None, r_in=None, r_out=None, r_out_erode=None):

        self._setting = setting
        self._pair_info = pair_info
        self._stage = stage
        self._padto = padto
        self._r_in = r_in
        self._r_out = r_out
        self._r_out_erode = r_out_erode

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
        self._win_center = np.array([0, 0, 0])
        self._sweep_completed = 0
        self._sweep_dim_completed = np.array([0, 0, 0])     # this flag is used when sweeping in one dimension is not needed. For instance if the
                                                            # if the size of the image is equal or smaller than the patch size. Only will be set in pad_for_sweeping()
        self._sweep_dim_end = np.array([0, 0, 0])

        # loading and preprocess images
        try:
            self._mask_to_zero = self._setting['network_dict']['stage' + str(self._stage)]['MaskToZero']
        except KeyError:
            self._mask_to_zero = None
            pass
        if fixed_im_sitk is None and moved_im_affine_sitk is None:
            self._fixed_im_sitk, self._moved_im_affine_sitk, fixed_mask_sitk, moved_mask_affine_sitk = self.load_pair()
        else:
            self._fixed_im_sitk = fixed_im_sitk
            self._moved_im_affine_sitk = moved_im_affine_sitk
        self._fixed_im = sitk.GetArrayFromImage(self._fixed_im_sitk)
        self._moved_im_affine = sitk.GetArrayFromImage(self._moved_im_affine_sitk)
        self._fixed_im_original_size = np.shape(self._fixed_im)

        if self._mask_to_zero:
            self._fixed_mask_sitk = fixed_mask_sitk
            self._fixed_mask = sitk.GetArrayFromImage(fixed_mask_sitk)
            self._moved_mask_affine_sitk = moved_mask_affine_sitk
            self._moved_mask_affine = sitk.GetArrayFromImage(moved_mask_affine_sitk)

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
            elx_all_points = elxpy.elxReadOutputPointsFile(su.address_generator(self._setting, 'Reg_BaseReg_OutputPoints',
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
                if min_coordinate_landmark[i] < self._r_in:
                    pad_before[i] = self._r_in - min_coordinate_landmark[i]
                if ((np.shape(self._fixed_im)[2-i] - max_coordinate_landmark[i]) - 1) < self._r_in:
                    pad_after[i] = self._r_in - (np.shape(self._fixed_im)[2-i] - max_coordinate_landmark[i]) + 1

            default_pixel_value = self._setting['data'][self._pair_info[0]['data']]['DefaultPixelValue']
            self._fixed_im = np.pad(self._fixed_im, ((pad_before[2], pad_after[2]), (pad_before[1], pad_after[1]), (pad_before[0], pad_after[0])),
                                    'constant', constant_values=(default_pixel_value,))
            self._moved_im_affine = np.pad(self._moved_im_affine, ((pad_before[2], pad_after[2]), (pad_before[1], pad_after[1]), (pad_before[0], pad_after[0])),
                                           'constant', constant_values=(default_pixel_value,))
            self._fixed_landmarks_index_padded = self._fixed_landmarks_index + pad_before

    def pad_for_sweeping(self):
        """
        This function pad the images in order to be used with a patch-based network or full-image network
        if padto is None: patch-based
            pad_before and pad_after is constant. In the cases that the input image is smaller than the patch size, we do extra padding
        else: full-image
            pad the input images to a fixed size.

        input: self._setting, self._fixed_im,
        return: self._extra_pad_before:
                self._extra_pad_after:
                self._offset_after_padding:
        """
        # if self._padto is None:
        pad_before = (self._r_in-self._r_out+self._r_out_erode)*np.ones(3, dtype=np.int16)     # zyx order
        pad_after = (self._r_in-self._r_out+self._r_out_erode)*np.ones(3, dtype=np.int16)      # zyx order
        size_after_padding = np.array(np.shape(self._fixed_im)) + pad_before + pad_after

        if (2 * self._r_in + 1) >= np.min(size_after_padding):
            for dim in range(0, 3):
                extra_pad = (size_after_padding[dim] - 2 * self._r_in - 1)
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
                        logging.debug('Extra uneven padding is done. pad_before[{}] = {}, pad_after[{}] = {} '.format(
                            dim, self._extra_pad_before[dim], dim, self._extra_pad_after[dim]))
                    else:
                        self._extra_pad_after[dim] = extra_pad_radius
                        pad_before[dim] = pad_before[dim] + self._extra_pad_before[dim]
                        pad_after[dim] = pad_after[dim] + self._extra_pad_after[dim]
                        logging.debug('Extra even_padding is done. pad_before[{}] = {}, pad_after[{}] = {} '.format(
                            dim, self._extra_pad_before[dim], dim, self._extra_pad_after[dim]))
                if extra_pad <= 0:  # we should consider both negative and zero extra_pad. which means that we cannot sweep in that direction anymore.
                    self._sweep_dim_completed[dim] = 1
                    self._sweep_dim_end[dim] = 1

        # else:
        #     dim_im = 3
        #     self._extra_pad_before = np.zeros(dim_im, dtype=np.int)
        #     self._extra_pad_after = np.zeros(dim_im, dtype=np.int)
        #     im_size = np.array(np.shape(self._fixed_im))
        #     extra_to_pad = self._padto - im_size
        #     for d in range(dim_im):
        #         if extra_to_pad[d] < 0:
        #             raise ValueError('size of the padto=' + str(self._padto) + ' should be smaller than the size of the image {}'.format(im_size))
        #         elif extra_to_pad[d] == 0:
        #             self._extra_pad_before[d] = 0
        #             self._extra_pad_after[d] = 0
        #         else:
        #             if extra_to_pad[d] % 2 == 0:
        #                 self._extra_pad_before[d] = np.int(extra_to_pad[d] / 2)
        #                 self._extra_pad_after[d] = np.int(extra_to_pad[d] / 2)
        #             else:
        #                 self._extra_pad_before[d] = np.floor(extra_to_pad[d] / 2)
        #                 self._extra_pad_after[d] = np.floor(extra_to_pad[d] / 2) + 1
        #
        #     pad_before = self._extra_pad_before.copy()
        #     pad_after = self._extra_pad_after.copy()
        #     self._sweep_dim_completed = [1, 1, 1]
        #     self._sweep_dim_end = [1, 1, 1]

        self._offset_after_padding = pad_before
        default_pixel_value = self._setting['data'][self._pair_info[0]['data']]['DefaultPixelValue']
        self._fixed_im = np.pad(self._fixed_im, ((pad_before[0], pad_after[0]),
                                                 (pad_before[1], pad_after[1]),
                                                 (pad_before[2], pad_after[2])),
                                'constant',
                                constant_values=(default_pixel_value,))
        self._moved_im_affine = np.pad(self._moved_im_affine, ((pad_before[0], pad_after[0]),
                                                               (pad_before[1], pad_after[1]),
                                                               (pad_before[2], pad_after[2])),
                                       'constant',
                                       constant_values=(default_pixel_value,))

    def preprocess_images(self):
        if self._setting['normalization'] == 'linear':
            self._fixed_im = ((self._fixed_im + 1000) / 4095.).astype(np.float32)
            self._moved_im_affine = ((self._moved_im_affine + 1000) / 4095.).astype(np.float32)
        if self._mask_to_zero is not None:
            default_pixel_value = self._setting['data'][self._pair_info[0]['data']]['DefaultPixelValue']
            self._fixed_im[self._fixed_mask == 0] = default_pixel_value
            self._moved_im_affine[self._moved_mask_affine == 0] = default_pixel_value

    def load_pair(self):
        base_reg = self._setting['BaseReg']
        im_info_fixed = copy.deepcopy(self._pair_info[0])
        im_info_fixed['stage'] = self._stage
        fixed_mask_sitk = None
        moved_mask_affine_sitk = None
        if 'dsmooth' in im_info_fixed:
            # in this case it means that images are synthetic
            im_info_su = {'data': im_info_fixed['data'], 'deform_exp': im_info_fixed['deform_exp'], 'type_im': im_info_fixed['type_im'],
                          'cn': im_info_fixed['cn'], 'dsmooth': im_info_fixed['dsmooth'], 'stage': self._stage, 'padto': im_info_fixed['padto']}
            fixed_im_address = su.address_generator(self._setting, 'DeformedIm', deformed_im_ext=im_info_fixed['deformed_im_ext'], **im_info_su)
            moved_im_affine_address = su.address_generator(self._setting, 'Im', **im_info_su)
            if self._mask_to_zero is not None:
                fixed_mask_address = su.address_generator(self._setting, 'Deformed' + self._mask_to_zero, **im_info_su)
                moved_mask_affine_address = su.address_generator(self._setting, self._mask_to_zero, **im_info_su)

        else:
            im_info_moving = copy.deepcopy(self._pair_info[1])
            im_info_moving['stage'] = self._stage
            fixed_im_address = su.address_generator(self._setting, 'Im', **im_info_fixed)
            moved_im_affine_address = su.address_generator(self._setting, 'MovedImBaseReg', pair_info=self._pair_info,
                                                           base_reg=base_reg, **im_info_moving)
            if self._mask_to_zero is not None:
                fixed_mask_address = su.address_generator(self._setting, self._mask_to_zero, **im_info_fixed)
                moved_mask_affine_address = su.address_generator(self._setting, 'Moved'+self._mask_to_zero+'BaseReg',
                                                                 pair_info=self._pair_info,  base_reg=base_reg, **im_info_moving)

        fixed_im_sitk = sitk.ReadImage(fixed_im_address)
        moved_im_affine_sitk = sitk.ReadImage(moved_im_affine_address)
        logging.info('FixedIm:'+fixed_im_address)
        logging.info('MovedImBaseReg:' + moved_im_affine_address)
        if self._mask_to_zero is not None:
            fixed_mask_sitk = sitk.ReadImage(fixed_mask_address)
            moved_mask_affine_sitk = sitk.ReadImage(moved_mask_affine_address)
            logging.info('FixedMask:' + fixed_mask_address)
            logging.info('MovedMaskBaseReg:' + moved_mask_affine_address)

        return fixed_im_sitk, moved_im_affine_sitk, fixed_mask_sitk, moved_mask_affine_sitk

    def next_landmark_patch(self, batch_size):
        raise RuntimeError('Deprecated')
        # if not self._chunks_completed:
        #     r = self._r_in
        #     ry = self._r_out
        #     endBatch = (self._batchCounter + 1) * batch_size
        #     if endBatch >= len(self._fixed_landmarks_index_padded):
        #         self._chunks_completed = 1
        #         endBatch = len(self._fixed_landmarks_index_padded)
        #     self._start_batch = self._batchCounter * batch_size
        #     self._end_batch = endBatch
        #     if self._setting['Dim'] == 3:
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

        r = self._r_in
        ry = self._r_out
        ry_erode = self._r_out_erode
        estimated_dvf_size = 2 * ry + 1

        if np.all(self._win_center == [0, 0, 0]):
            self._win_center = [r, r, r]
        if self._setting['Dim'] == 3:
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
        predicted_end = estimated_dvf_size - ry_erode * np.ones(3, dtype=np.int16) - self._extra_pad_after

        if self._setting['verbose']:
            logging.debug(self._pair_info[0]['data'] + ', CN{}, win_center={} '.format(
                self._pair_info[0]['cn'], self._win_center))

        if self._sweep_dim_end[0] & self._sweep_dim_end[1]:
            self._win_center[0] = r
            self._win_center[1] = r
            self._win_center[2] = self._win_center[2] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[2]+r+1 > np.shape(self._fixed_im)[2]:
                self._win_center[2] = np.shape(self._fixed_im)[2] - r - 1
                if not self._sweep_dim_end[2]:
                    if not self._sweep_dim_completed[0]:
                        self._sweep_dim_end[0] = 0
                    if not self._sweep_dim_completed[1]:
                        self._sweep_dim_end[1] = 0
                    self._sweep_dim_end[2] = 1
                else:
                    self._sweep_completed = 1
                    logging.debug('stage{} sweep is completed'.format(self._stage))
            else:
                self._sweep_dim_end[0] = 0
                self._sweep_dim_end[1] = 0
        elif self._sweep_dim_end[0]:
            self._win_center[0] = r
            self._win_center[1] = self._win_center[1] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[1]+r+1 > np.shape(self._fixed_im)[1]:
                self._win_center[1] = np.shape(self._fixed_im)[1] - r - 1
                self._sweep_dim_end[1] = 1
            if not self._sweep_dim_completed[0]:
                self._sweep_dim_end[0] = 0
        else:
            self._win_center[0] = self._win_center[0] + 2 * ry - 2 * ry_erode + 1
            if self._win_center[0]+r+1 > np.shape(self._fixed_im)[0]:
                self._win_center[0] = np.shape(self._fixed_im)[0] - r - 1
                self._sweep_dim_end[0] = 1

        batch_both = np.concatenate((batch_moved_affine, batch_fixed), axis=4)
        if self._setting['verbose'] and not self._sweep_completed:
            logging.debug(self._pair_info[0]['data'] + ', CN{} next_sweep_patch: win_center={} '.format(
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

    def get_fixed_mask_sitk(self):
        return self._fixed_mask_sitk

    def get_moved_mask_affine_sitk(self):
        return self._moved_mask_affine_sitk

    def get_sweep_completed(self):
        return self._sweep_completed
