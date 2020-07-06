import copy
import os
import numpy as np
import SimpleITK as sitk
import functions.image.image_processing as ip


def img_converter(data_folder, data, type_im, cn, ext='.mha', mha_folder_name='mha', point_folder_name='points'):
    """
    convert img images to mha.
    reading image:
    1) Size and voxel spacing of the images are available at https://www.dir-lab.com/ReferenceData.html
    2) The superior-inferior axis needs to be flipped
    3) Empty slices will be removed
        copd1_eBHCT.mha slice 0:1
        copd2_eBHCT.mha slice 0:6
        copd3_eBHCT.mha slice 0:9
        copd4_eBHCT.mha slice 0:9
        copd5_eBHCT.mha slice NA
        copd6_eBHCT.mha slice 0:2
        copd7_eBHCT.mha slice 0:9
        copd8_eBHCT.mha slice 0:7
        copd9_eBHCT.mha slice 0:19
        copd10_iBHCT.mah slice 0
    4) Index modification:
        4a) The superior-inferior axis are flipped. The reason is that to make it more similar to SPREAD study.
        4b) The indices start at 1. We like them to start at 0.
        4c) change indices of landmarks based on the removed slices

    5) Normalize the intensity by subtracting by -1024
    6) Set the outside value to -2048

    :param ext
    :param cn:
    :param type_im
    :param data_folder
    :param data
    :param mha_folder_name
    :param point_folder_name
    :return: converted mha image and converted landmark files:
        example: 
            copd1_eBHCT.mha
            copd1_300_eBH_world_r1_tr.txt:  landmarks in world coordinate (truncated)
            copd1_300_eBH_world_r1_elx.txt: landmarks in world coordinate with two additional lines for elastix
            copd1_300_eBH_xyz_r1_tr.txt:    landmark in indices
            copd1_300_eBH_xyz_r1_elx.txt:   landmark in indices with two additional lines for elastix
    """

    if data == 'DIR-Lab_4D':
        type_im_list = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
        data_folder_sub = data_folder + 'DIR-Lab/4DCT/'
        if cn < 6:
            im_img_name = 'Images/case' + str(cn) + '_' + type_im_list[type_im] + '_s.img'
        else:
            im_img_name = 'Images/case' + str(cn) + '_' + type_im_list[type_im] + '.img'
        im_img_folder = data_folder_sub + 'Case' + str(cn) + 'Pack/'
        if cn == 8:
            im_img_folder = data_folder_sub + 'Case' + str(cn) + 'Deploy/'
        im_mha_name = 'case' + str(cn) + '_' + type_im_list[type_im] + ext
        im_mha_folder = data_folder_sub + mha_folder_name + '/case' + str(cn) + '/'
        point_folder = data_folder_sub + point_folder_name + '/case' + str(cn) + '/'
        if cn < 6:
            index_tr_old_address_list = [im_img_folder + '/Sampled4D/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '.txt',
                                         im_img_folder + '/ExtremePhases/case' + str(cn) + '_300_' + type_im_list[type_im] + '_xyz.txt']
        else:
            index_tr_old_address_list = [im_img_folder + '/Sampled4D/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '.txt',
                                         im_img_folder + '/ExtremePhases/case' + str(cn) + '_dirLab300_' + type_im_list[type_im] + '_xyz.txt']
        index_tr_new_address_list = [point_folder + '/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '_xyz_tr.txt',
                                     point_folder + '/case' + str(cn) + '_300_' + type_im_list[type_im] + '_xyz_tr.txt']
        index_elx_new_address_list = [point_folder + '/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '_xyz_elx.txt',
                                      point_folder + '/case' + str(cn) + '_300_' + type_im_list[type_im] + '_xyz_elx.txt']
        point_tr_new_address_list = [point_folder + '/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '_world_tr.txt',
                                     point_folder + '/case' + str(cn) + '_300_' + type_im_list[type_im] + '_world_tr.txt']
        point_elx_new_address_list = [point_folder + '/case' + str(cn) + '_4D-75_' + type_im_list[type_im] + '_world_elx.txt',
                                      point_folder + '/case' + str(cn) + '_300_' + type_im_list[type_im] + '_world_elx.txt']
        dirlab_header = dirlab_4dct_header()

    elif data == 'DIR-Lab_COPD':
        type_im_list = ['i', 'e']
        data_folder_sub = data_folder + 'DIR-Lab/COPDgene/'
        im_img_name = 'copd' + str(cn) + '_' + type_im_list[type_im] + 'BHCT.img'
        im_img_folder = data_folder_sub + 'copd' + str(cn) + '/'
        im_mha_name = 'copd' + str(cn) + '_' + type_im_list[type_im] + 'BHCT' + ext
        im_mha_folder = data_folder_sub + mha_folder_name + '/'
        point_folder = data_folder_sub + point_folder_name
        index_tr_old_address_list = [im_img_folder + 'copd' + str(cn) + '_300_' + type_im_list[type_im] + 'BH_xyz_r1.txt']
        index_tr_new_address_list = [point_folder + '/copd' + str(cn) + '_300_' + type_im_list[type_im] + 'BH_xyz_r1_tr.txt']
        index_elx_new_address_list = [point_folder + '/copd' + str(cn) + '_300_' + type_im_list[type_im] + 'BH_xyz_r1_elx.txt']
        point_tr_new_address_list = [point_folder + '/copd' + str(cn) + '_300_' + type_im_list[type_im] + 'BH_world_r1_tr.txt']
        point_elx_new_address_list = [point_folder + '/copd' + str(cn) + '_300_' + type_im_list[type_im] + 'BH_world_r1_elx.txt']
        dirlab_header = dirlab_copd_header()

    else:
        raise ValueError('Data=' + data + ", it should be in ['DIR-Lab_4D', 'DIR-Lab_COPD']")

    if not os.path.isdir(im_mha_folder):
        os.makedirs(im_mha_folder)
    if not os.path.isdir(point_folder):
        os.makedirs(point_folder)
    im_img_address = im_img_folder + im_img_name
    im_mha_address = im_mha_folder + im_mha_name
    if not os.path.isfile(im_mha_address):
        # 1,2) reading image:----------------------------------------------------------------
        fid = open(im_img_address, 'rb')
        im_data = np.fromfile(fid, np.int16)
        image_old = im_data.reshape(dirlab_header['case' + str(cn)]['Size'][::-1])
        image_old = np.flip(image_old, axis=0)  # The superior-inferior axis needs to be flipped
        origin = [0, 0, 0]
        image = copy.deepcopy(image_old)
        # reading landmarks:
        for ii, index_tr_old_address in enumerate(index_tr_old_address_list):
            index_tr_new_address = index_tr_new_address_list[ii]
            index_elx_new_address = index_elx_new_address_list[ii]
            point_tr_new_address = point_tr_new_address_list[ii]
            point_elx_new_address = point_elx_new_address_list[ii]
            if os.path.isfile(index_tr_old_address):
                index_tr_old_raw = np.loadtxt(index_tr_old_address)
                # 4a&b) The superior-inferior axis is flipped. be careful about that indices start at 1. after converting to zero-start,
                #  there is no -1 in the SI direction.

                index_tr_old = np.array([[index_tr_old_raw[i, 0] - 1,
                                          index_tr_old_raw[i, 1] - 1,
                                          image_old.shape[0] - index_tr_old_raw[i, 2]]
                                         for i in range(index_tr_old_raw.shape[0])])

                # 3) remove empty slices only in DIR-Lab_COPD-----------------------------------------
                if data == 'DIR-Lab_COPD':
                    image, slices_to_remove = remove_empty_slices(image_old)
                    print(im_img_name + ' slices are removed: ' + str(slices_to_remove))
                    shift_indices = len(slices_to_remove)
                    shift_world = shift_indices * dirlab_header['case' + str(cn)]['Spacing'][2]
                    origin[2] = shift_world

                    # 4c) change indices of landmarks based on the removed slices
                    index_tr_new = [[index_tr_old[i, 0], index_tr_old[i, 1], index_tr_old[i, 2] - shift_indices] for i in range(index_tr_old.shape[0])]
                else:
                    index_tr_new = index_tr_old.copy()

                np.savetxt(index_tr_new_address, index_tr_new, fmt='%d')
                point_tr_new = ip.index_to_world(index_tr_new, spacing=dirlab_header['case' + str(cn)]['Spacing'], origin=origin)
                np.savetxt(point_tr_new_address, point_tr_new, fmt='%-9.3f')
                open_text = open(index_tr_new_address, "r")
                number_of_landmarks = index_tr_new.shape[0]
                with open(index_elx_new_address, "w") as open_elx:
                    open_elx.write('index \n')
                    open_elx.write(str(number_of_landmarks) + ' \n')
                    open_elx.write(open_text.read())
                open_text.close()

                open_text = open(point_tr_new_address, "r")
                with open(point_elx_new_address, "w") as open_elx:
                    open_elx.write('point \n')
                    open_elx.write(str(number_of_landmarks) + ' \n')
                    open_elx.write(open_text.read())
                open_text.close()

        # 5) normalize the intensity
        image = image - 1024  # we are not sure about the slope and intercept.

        # 6) set the outside value to -2048
        image[image == -3024] = -2048
        image_sitk = ip.array_to_sitk(image, spacing=dirlab_header['case' + str(cn)]['Spacing'], origin=origin)
        sitk.WriteImage(image_sitk, im_mha_address)
        print('case' + str(cn) + ' type' + str(type_im) + ' is done..')


def remove_empty_slices(image):
    slices_to_remove = []
    for slice_index in range(np.shape(image)[0]):
        if np.sum(image[slice_index, :, :]) == 0:
            slices_to_remove.append(slice_index)
    slices_all = [i for i in range(np.shape(image)[0])]
    slices_to_keep = [i for i in slices_all if i not in slices_to_remove]
    image_cropped = image[slices_to_keep, :, :]

    return image_cropped, slices_to_remove


def dirlab_copd_header():
    """
    size and voxel spacing of the images are available at https://www.dir-lab.com/ReferenceData.html
    """
    dirlab_info = dict()
    for cn in range(1, 11):
        dirlab_info['case' + str(cn)] = {}
    dirlab_info['case1']['Size'] = [512, 512, 121]
    dirlab_info['case2']['Size'] = [512, 512, 102]
    dirlab_info['case3']['Size'] = [512, 512, 126]
    dirlab_info['case4']['Size'] = [512, 512, 126]
    dirlab_info['case5']['Size'] = [512, 512, 131]
    dirlab_info['case6']['Size'] = [512, 512, 119]
    dirlab_info['case7']['Size'] = [512, 512, 112]
    dirlab_info['case8']['Size'] = [512, 512, 115]
    dirlab_info['case9']['Size'] = [512, 512, 116]
    dirlab_info['case10']['Size'] = [512, 512, 135]

    dirlab_info['case1']['Spacing'] = [0.625, 0.625, 2.5]
    dirlab_info['case2']['Spacing'] = [0.645, 0.645, 2.5]
    dirlab_info['case3']['Spacing'] = [0.652, 0.652, 2.5]
    dirlab_info['case4']['Spacing'] = [0.590, 0.590, 2.5]
    dirlab_info['case5']['Spacing'] = [0.647, 0.647, 2.5]
    dirlab_info['case6']['Spacing'] = [0.633, 0.633, 2.5]
    dirlab_info['case7']['Spacing'] = [0.625, 0.625, 2.5]
    dirlab_info['case8']['Spacing'] = [0.586, 0.586, 2.5]
    dirlab_info['case9']['Spacing'] = [0.644, 0.644, 2.5]
    dirlab_info['case10']['Spacing'] = [0.742, 0.742, 2.5]

    return dirlab_info


def dirlab_4dct_header():
    """
    size and voxel spacing of the images are available at https://www.dir-lab.com/ReferenceData.html
    """
    dirlab_info = dict()
    for cn in range(1, 11):
        dirlab_info['case' + str(cn)] = {}
    dirlab_info['case1']['Size'] = [256, 256, 94]
    dirlab_info['case2']['Size'] = [256, 256, 112]
    dirlab_info['case3']['Size'] = [256, 256, 104]
    dirlab_info['case4']['Size'] = [256, 256, 99]
    dirlab_info['case5']['Size'] = [256, 256, 106]
    dirlab_info['case6']['Size'] = [512, 512, 128]
    dirlab_info['case7']['Size'] = [512, 512, 136]
    dirlab_info['case8']['Size'] = [512, 512, 128]
    dirlab_info['case9']['Size'] = [512, 512, 128]
    dirlab_info['case10']['Size'] = [512, 512, 120]

    dirlab_info['case1']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case2']['Spacing'] = [1.16, 1.16, 2.5]
    dirlab_info['case3']['Spacing'] = [1.15, 1.15, 2.5]
    dirlab_info['case4']['Spacing'] = [1.13, 1.13, 2.5]
    dirlab_info['case5']['Spacing'] = [1.10, 1.10, 2.5]
    dirlab_info['case6']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case7']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case8']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case9']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case10']['Spacing'] = [0.97, 0.97, 2.5]

    return dirlab_info


def main():
    data = 'DIR-Lab_4D'
    data_folder = 'E:/PHD/Database/'
    for cn in range(6, 11):
        for type_im in [0, 5]:
            img_converter(data_folder=data_folder, data=data, type_im=type_im, cn=cn)


if __name__ == '__main__':
    main()
