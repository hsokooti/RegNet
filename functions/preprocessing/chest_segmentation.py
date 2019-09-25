import os
import sys
import numpy as np
import functions.setting.setting_utils as su
import SimpleITK as sitk
import scipy.ndimage as ndimage
import functions.image.image_processing as ip


def ptpulmo_segmentation(segment_organ, data_folder, data, type_im, cn, ext='.mha', mha_folder_name='mha', ptpulmo_exe=None, ptpulmu_setting=None):

    if segment_organ == 'Torso':
        organ_cmd = 'torso'
        output_name = 'Torso'
        segment_folder_name = 'Torso'

    elif segment_organ == 'Lung':
        organ_cmd = 'lungB'
        output_name = 'Lung_both'
        segment_folder_name = 'Lung_Initial'

    else:
        raise ValueError('segment_organ is ' + segment_organ + '. However it should be "Troso" or "Lung" ')

    if ptpulmo_exe is None:
        ptpulmo_exe = 'E:/PHD/Software/Works_Other/LungSegmentation/ptpulmo.exe'
    if ptpulmu_setting is None:
        ptpulmu_setting = 'E:/PHD/Software/Works_Other/LungSegmentation/PulmoDefaultSettings.psf'

    if data == 'DIR-Lab_4D':
        type_im_list = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
        data_folder_sub = data_folder + 'DIR-Lab/4DCT/'
        im_address = data_folder_sub+mha_folder_name+'/case'+str(cn)+'/'+'case'+str(cn)+'_'+type_im_list[type_im]+ext
        segmented_folder = data_folder_sub+mha_folder_name+'/case'+str(cn)+'/'+segment_folder_name+'/'
        segmented_address = segmented_folder+'case'+str(cn)+'_'+type_im_list[type_im]+'_'+segment_folder_name+ext

    elif data == 'DIR-Lab_COPD':
        type_im_list = ['i', 'e']
        data_folder_sub = data_folder + 'DIR-Lab/COPDgene/'
        im_address = data_folder_sub+mha_folder_name+'/copd'+str(cn)+'_'+type_im_list[type_im]+'BHCT'+ext
        segmented_folder = data_folder_sub+mha_folder_name+'/'+segment_folder_name+'/'
        segmented_address = segmented_folder+'copd'+str(cn)+'_'+type_im_list[type_im]+'BHCT'+ext

    else:
        raise ValueError('Data='+data+", it should be in ['DIR-Lab_4D', 'DIR-Lab_COPD']")

    if not os.path.isdir(segmented_folder):
        os.makedirs(segmented_folder)

    if not os.path.isfile(segmented_address):
        segmentation_cmd = "%s -in %s -ps %s -out %s -%s" % (ptpulmo_exe, im_address, ptpulmu_setting, segmented_folder, organ_cmd)
        if not os.system(segmentation_cmd) == 0:
            print("ptpulmo failed")
            sys.exit(1)
        else:
            torso_old_address = segmented_folder + output_name + '.mha'
            os.replace(torso_old_address, segmented_address)


def lung_fill_hole(data_folder, data, type_im, cn, ext='.mha', mha_folder_name='mha'):
    lung_initial_folder_name = 'Lung_Initial'
    lung_filled_folder_name = 'Lung_Filled'
    if data == 'DIR-Lab_4D':
        type_im_list = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
        data_folder_sub = data_folder + 'DIR-Lab/4DCT/'
        lung_initial_folder = data_folder_sub+mha_folder_name+'/case'+str(cn)+'/'+lung_initial_folder_name+'/'
        lung_filled_folder = data_folder_sub+mha_folder_name+'/case'+str(cn)+'/'+lung_filled_folder_name+'/'
        lung_initial_address = lung_initial_folder+'case'+str(cn)+'_'+type_im_list[type_im]+'_'+lung_initial_folder_name+ext
        lung_filled_address = lung_filled_folder+'case'+str(cn)+'_'+type_im_list[type_im]+'_'+lung_filled_folder_name+ext

    elif data == 'DIR-Lab_COPD':
        type_im_list = ['i', 'e']
        data_folder_sub = data_folder + 'DIR-Lab/COPDgene/'
        lung_initial_folder = data_folder_sub+mha_folder_name+'/'+lung_initial_folder_name+'/'
        lung_filled_folder = data_folder_sub+mha_folder_name+'/'+lung_filled_folder_name+'/'
        lung_initial_address = lung_initial_folder+'copd'+str(cn)+'_'+type_im_list[type_im]+'BHCT'+ext
        lung_filled_address = lung_filled_folder+'copd'+str(cn)+'_'+type_im_list[type_im]+'BHCT'+ext

    else:
        raise ValueError('Data='+data+", it should be in ['DIR-Lab_4D', 'DIR-Lab_COPD']")

    if not os.path.isdir(lung_filled_folder):
        os.makedirs(lung_filled_folder)

    lung_initial_sitk = sitk.ReadImage(lung_initial_address)
    lung_initial = sitk.GetArrayFromImage(lung_initial_sitk)
    lung_initial = lung_initial > 0
    structure = np.ones([3, 3, 3], dtype=np.bool)
    lung_filled = (ndimage.morphology.binary_closing(lung_initial, structure=structure)).astype(np.int8)
    sitk.WriteImage(ip.array_to_sitk(lung_filled, im_ref=lung_initial_sitk), lung_filled_address)


def lung_fill_hole_erode(setting, cn=None):
    folder = su.address_generator(setting, 'Lung_Filled_Erode', cn=cn, type_im=0).rsplit('/', maxsplit=1)[0]
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for type_im in range(len(setting['types'])):
        lung_raw_filled_sitk = sitk.ReadImage(su.address_generator(setting, 'Lung_Filled', cn=cn, type_im=type_im))
        lung_raw_filled = sitk.GetArrayFromImage(lung_raw_filled_sitk)

        lung_raw_filled = lung_raw_filled > 0
        structure = np.ones([3, 3, 3], dtype=np.bool)
        lung_filled_erode = (ndimage.morphology.binary_dilation(lung_raw_filled, structure=structure)).astype(np.int8)
        sitk.WriteImage(ip.array_to_sitk(lung_filled_erode, im_ref=lung_raw_filled_sitk),
                        su.address_generator(setting, 'Lung_Filled_Erode', cn=cn, type_im=type_im))


def main():
    data = 'DIR-Lab_4D'
    data_folder = 'E:/PHD/Database/'
    for cn in range(1, 11):
        for type_im in range(0, 10):
            ptpulmo_segmentation(segment_organ='Lung', data_folder=data_folder, data=data, type_im=type_im, cn=cn)
            lung_fill_hole(data_folder=data_folder, data=data, type_im=type_im, cn=cn)


if __name__ == '__main__':
    main()
