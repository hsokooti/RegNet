import numpy as np
import SimpleITK as sitk
import os
import functions.setting.setting_utils as su
import functions.image.image_processing as ip
import scipy.ndimage as ndimage
import logging


def cylinder_mask(setting, cn=None, overwrite=False):
    cylinder_folder = su.address_generator(setting, 'Cylinder', cn=cn, type_im=0).rsplit('/', maxsplit=1)[0]
    if not os.path.isdir(cylinder_folder):
        os.makedirs(cylinder_folder)
    for type_im in range(len(setting['types'])):
        cylinder_mask_address = su.address_generator(setting, 'Cylinder', cn=cn, type_im=type_im)
        if (not os.path.isfile(cylinder_mask_address)) or overwrite:
            image_sitk = sitk.ReadImage(su.address_generator(setting, 'Im', cn=cn, type_im=type_im))
            cylinder_mask_sitk = sitk.BinaryThreshold(image_sitk,
                                                      lowerThreshold=setting['DefaultPixelValue']-1,
                                                      upperThreshold=setting['DefaultPixelValue']+0.01,
                                                      insideValue=0,
                                                      outsideValue=1)
            structure = np.ones((1, 3, 3))
            # erosion with ndimage is 5 times faster than SimpleITK
            cylinder_mask_eroded = (ndimage.binary_erosion(sitk.GetArrayFromImage(cylinder_mask_sitk), structure=structure, iterations=2)).astype(np.int8)
            cylinder_mask_eroded_sitk = ip.array_to_sitk(cylinder_mask_eroded, im_ref=image_sitk)
            sitk.WriteImage(cylinder_mask_eroded_sitk, cylinder_mask_address)
            logging.debug(cylinder_mask_address + ' is done')
