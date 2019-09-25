from .. import network


def find_optimal_radius(im_sitk, current_network_name, r_out_erode_default, gpu_memory=None, number_of_gpu=None):
    """
    First set the optimal size to the maximum radius possible with respect to GPU memory.
    Then decrease it as much as you can, but not smaller than the image size nor the radius used in training
    :param im_sitk:
    :param current_network_name:
    :param r_out_erode_default:
    :param gpu_memory:
    :param number_of_gpu:
    :return:
    """
    r_in_max, r_out_max = getattr(getattr(network, current_network_name), 'maximum_radius_test')(gpu_memory, number_of_gpu)
    r_in_min, r_out_min = getattr(getattr(network, current_network_name), 'raidus_train')()
    resize_unit = getattr(getattr(network, current_network_name), 'get_resize_unit')()
    r_in, r_out = None, None

    im_size_max = max(im_sitk.GetSize())
    r_image_max = im_size_max // 2
    resize = 0

    r_out_erode = r_out_erode_default

    if r_image_max <= r_out_min:
        r_in = r_in_min
        r_out = r_out_min
        r_out_erode = 0

    elif r_image_max <= r_out_max:
        r_out_erode = 0
        while (r_out_max - (resize + resize_unit)) >= r_image_max and (r_out_max - (resize + resize_unit)) >= r_out_min:
            resize = resize + resize_unit

    if r_in is None or r_out is None:
        r_in = r_in_max - resize
        r_out = r_out_max - resize

    return r_in, r_out, r_out_erode
