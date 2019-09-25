import os
import functions.setting.setting_utils as su


def seed_number_by_im_info(im_info, string_exp, stage=1, gonna_generate_next_im=False):
    seed_number = sum([ord(i) for i in string_exp]) * 1000 + sum([ord(i) for i in im_info['data']]) * 501 + \
                  im_info['type_im'] * 347 + im_info['cn'] * 231 + im_info['dsmooth'] * 26 + stage * 11 + gonna_generate_next_im * 2
    return seed_number


def remove_redundant_images(setting, im_info, stage=1):
    """
    Remove DeformedDVF and DeformedImage and NextIm from stage 1
    :param setting:
    :param im_info:
    :param stage:
    :return:
    """

    im_info_su = {'data': im_info['data'], 'deform_exp': im_info['deform_exp'], 'type_im': im_info['type_im'],
                  'cn': im_info['cn'], 'dsmooth': im_info['dsmooth'], 'stage': stage}

    im_list_remove = list()
    im_list_remove.append(su.address_generator(setting, 'DeformedDVF', **im_info_su))
    im_list_remove.append(su.address_generator(setting, 'Jac', **im_info_su))

    deformed_im_ext_combined = []
    for i_ext, deformed_im_ext_current in enumerate(im_info['deformed_im_ext']):
        deformed_im_ext_combined.append(deformed_im_ext_current)
        deformed_im_address = su.address_generator(setting, 'DeformedIm', deformed_im_ext=deformed_im_ext_combined, **im_info_su)
        im_list_remove.append(deformed_im_address)

    im_list_remove.append(su.address_generator(setting, 'DeformedTorso', **im_info_su))
    im_list_remove.append(su.address_generator(setting, 'NextIm', **im_info_su))
    im_list_remove.append(su.address_generator(setting, 'NextTorso', **im_info_su))
    im_list_remove.append(su.address_generator(setting, 'NextLung', **im_info_su))

    for im_address in im_list_remove:
        if os.path.isfile(im_address):
            os.remove(im_address)
