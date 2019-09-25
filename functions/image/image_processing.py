import math
import numpy as np
import os
import SimpleITK as sitk
import time
import functions.kernel.conv_kernel as conv_kernel
import functions.tf_utils as tfu


def ReadImage(im_address, waiting_time=1):
    """
    simpleITK when writing, creates a blank file then fills it within few seconds. This waiting prevents reading blank files.
    :param im_address:
    :param waiting_time:
    :return:
    """
    while (time.time() - os.path.getmtime(im_address)) < waiting_time:
        time.sleep(1)
    im_sitk = sitk.ReadImage(im_address)
    return im_sitk


def calculate_jac(dvf, voxel_size=None):
    """
    :param dvf: a numpy array with shape of (sizeY, sizeX, 2) or (sizeZ, sizeY, sizeX, 3). You might use np.transpose before this function to correct the order of DVF shape.
    :param voxel_size: physical voxel spacing in mm
    :return: Jac
    """

    if voxel_size is None:
        voxel_size = [1, 1, 1]
    if (len(np.shape(dvf)) - 1) != len(voxel_size):
        raise ValueError('dimension of DVF is {} but dimension of voxelSize is {}'.format(
            len(np.shape(dvf)) - 1, len(voxel_size)))
    T = np.zeros(np.shape(dvf), dtype=np.float32) # derivative should be calculated on T which is DVF + indices (world coordinate)
    indices = [None] * (len(np.shape(dvf)) - 1)
    dvf_grad = []

    if len(voxel_size) == 2:
        indices[0], indices[1] = np.meshgrid(np.arange(0, np.shape(dvf)[0]),
                                             np.arange(0, np.shape(dvf)[1]),
                                             indexing='ij')
    if len(voxel_size) == 3:
        indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, np.shape(dvf)[0]),
                                                         np.arange(0, np.shape(dvf)[1]),
                                                         np.arange(0, np.shape(dvf)[2]),
                                                         indexing='ij')

    for d in range(len(voxel_size)):
        indices[d] = indices[d] * voxel_size[d]
        T[:, :, :, d] = dvf[:, :, :, d] + indices[d]
        dvf_grad.append([grad_mat / voxel_size[d] for grad_mat in np.gradient(T[:, :, :, d])])  # DVF.grad can be calculated in one shot without for loop.
    if len(voxel_size) == 2:
        jac = dvf_grad[0][0] * dvf_grad[1][1] - dvf_grad[0][1] * dvf_grad[1][0]
        #       f0/dir0      *   f1/dir1      -    f0/dir1     *   f1/dir0

    elif len(voxel_size) == 3:
        jac = (dvf_grad[0][0] * dvf_grad[1][1] * dvf_grad[2][2] +  # f0/dir0 + f1/dir1 + f2/dir2
               dvf_grad[0][1] * dvf_grad[1][2] * dvf_grad[2][0] +  # f0/dir1 + f1/dir2 + f2/dir0
               dvf_grad[0][2] * dvf_grad[1][0] * dvf_grad[2][1] -
               dvf_grad[0][2] * dvf_grad[1][1] * dvf_grad[2][0] -
               dvf_grad[0][1] * dvf_grad[1][0] * dvf_grad[2][2] -
               dvf_grad[0][0] * dvf_grad[1][2] * dvf_grad[2][1]
               )
    else:
        raise ValueError('Length of voxel size should be 2 or 3')
    return jac


def resampler_by_transform(im_sitk, dvf_t, im_ref=None, default_pixel_value=0, interpolator=sitk.sitkBSpline):
    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)
    out_im = resampler.Execute(im_sitk)
    return out_im


def array_to_sitk(array_input, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None):
    if origin is None:
        origin = [0, 0, 0]
    if spacing is None:
        spacing = [1, 1, 1]
    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    sitk_output = sitk.GetImageFromArray(array_input, isVector=is_vector)
    if im_ref is None:
        sitk_output.SetOrigin(origin)
        sitk_output.SetSpacing(spacing)
        sitk_output.SetDirection(direction)
    else:
        sitk_output.SetOrigin(im_ref.GetOrigin())
        sitk_output.SetSpacing(im_ref.GetSpacing())
        sitk_output.SetDirection(im_ref.GetDirection())
    return sitk_output


def upsampler_gpu(input, up_scale, output_shape_3d=None):
    """
    Upsampling wiht GPU by an integer scale
    :param input: can be a 3D numpy array or sitk image
    :param up_scale: an integer value!
    :param output_shape_3d:
    :return: output: can be a numpy array or sitk image based on the input
    """
    import tensorflow as tf
    if isinstance(input, sitk.Image):
        input_numpy = sitk.GetArrayFromImage(input)
        mode = 'sitk'
    else:
        input_numpy = input
        mode = 'numpy'
    if not isinstance(up_scale, int):
        raise ValueError('upscale should be integer. now it is {} with type of '.format(str(up_scale)) + type(up_scale))

    orginal_input_shape = np.shape(input_numpy)
    if len(orginal_input_shape) < 3:
        input_numpy = np.expand_dims(input_numpy, -1)

    tf.reset_default_graph()
    sess = tf.Session()
    input_tf = tf.placeholder(tf.float32, shape=[1, None, None, None, np.shape(input_numpy)[3]], name="Input")
    upsampled_tf = tfu.layers.upsampling3d(input_tf, 'UpSampling', scale=up_scale, interpolator='trilinear', padding_mode='SYMMETRIC',
                                           padding='same', output_shape_3d=output_shape_3d)

    # upsampled_tf = tf.squeeze(upsampled_batch_tf, axis=0)
    sess.run(tf.global_variables_initializer())
    [upsampled_numpy] = sess.run([upsampled_tf], feed_dict={input_tf: np.expand_dims(input_numpy, axis=0)})

    upsampled_numpy = np.squeeze(upsampled_numpy, 0)
    if len(orginal_input_shape) < 3:
        upsampled_numpy = np.squeeze(upsampled_numpy, -1)

    if mode == 'numpy':
        output = upsampled_numpy
    elif mode == 'sitk':
        output = array_to_sitk(upsampled_numpy.astype(np.float64),
                               origin=input.GetOrigin(),
                               spacing=tuple(i / up_scale for i in input.GetSpacing()),
                               direction=input.GetDirection(),
                               is_vector=True)
    else:
        output = None

    return output


def upsampler_gpu_old(input, up_scale, default_pixel_value=0, dvf_output_size=None):
    """
    Upsampling wiht GPU by an integer scale
    :param input: can be a 3D numpy array or sitk image
    :param up_scale: an integer value!
    :param default_pixel_value:
    :return: output: can be a numpy array or sitk image based on the input
    """
    import tensorflow as tf
    if isinstance(input, sitk.Image):
        input_numpy = sitk.GetArrayFromImage(input)
        mode = 'sitk'
    else:
        input_numpy = input
        mode = 'numpy'
    if not isinstance(up_scale, int):
        raise ValueError('upscale should be integer. now it is {} with type of '.format(str(up_scale)) + type(up_scale))

    tf.reset_default_graph()
    sess = tf.Session()
    dvf_tf = tf.placeholder(tf.float32, shape=[1, None, None, None, 3], name="DVF_Input")
    DVF_outSize = tf.placeholder(tf.int32, shape=[3], name='DVF_outSize')
    convKernelBiLinear = conv_kernel.bilinear_up_kernel(dim=3)
    convKernelBiLinear = np.expand_dims(convKernelBiLinear, -1)
    convKernelBiLinear = np.expand_dims(convKernelBiLinear, -1)
    convKernelBiLinear = tf.constant(convKernelBiLinear)
    myDVF0 = tf.expand_dims(dvf_tf[:, :, :, :, 0], -1)
    myDVF1 = tf.expand_dims(dvf_tf[:, :, :, :, 1], -1)
    myDVF2 = tf.expand_dims(dvf_tf[:, :, :, :, 2], -1)
    upSampledDVF0 = tf.nn.conv3d_transpose(myDVF0, convKernelBiLinear, output_shape=(1, DVF_outSize[0], DVF_outSize[1], DVF_outSize[2], 1), strides=(1, up_scale, up_scale, up_scale, 1))
    upSampledDVF1 = tf.nn.conv3d_transpose(myDVF1, convKernelBiLinear, output_shape=(1, DVF_outSize[0], DVF_outSize[1], DVF_outSize[2], 1), strides=(1, up_scale, up_scale, up_scale, 1))
    upSampledDVF2 = tf.nn.conv3d_transpose(myDVF2, convKernelBiLinear, output_shape=(1, DVF_outSize[0], DVF_outSize[1], DVF_outSize[2], 1), strides=(1, up_scale, up_scale, up_scale, 1))
    upSampledDVF = tf.squeeze(tf.concat([upSampledDVF0, upSampledDVF1, upSampledDVF2], -1), axis=0)
    sess.run(tf.global_variables_initializer())
    [output_numpy] = sess.run([upSampledDVF], feed_dict={dvf_tf: np.expand_dims(input_numpy, axis=0),
                                                         DVF_outSize: dvf_output_size})
    if mode == 'numpy':
        output = output_numpy
    elif mode == 'sitk':
        output = array_to_sitk(output_numpy.astype(np.float64),
                               origin=input.GetOrigin(),
                               spacing=tuple(i / up_scale for i in input.GetSpacing()),
                               direction=input.GetDirection(),
                               is_vector=True)
    return output


def downsampler_gpu(input, down_scale, kernel_name='bspline', normalize_kernel=True, a=-0.5, default_pixel_value=0):
    """
    Downsampling wiht GPU by an integer scale
    :param input: can be a 2D or 3D numpy array or sitk image
    :param down_scale: an integer value!
    :param kernel_name:
    :param normalize_kernel:
    :param a:
    :param default_pixel_value:
    :return: output: can be a numpy array or sitk image based on the input
    """
    import tensorflow as tf
    if isinstance(input, sitk.Image):
        input_numpy = sitk.GetArrayFromImage(input)
        mode = 'sitk'
    else:
        input_numpy = input
        mode = 'numpy'
    if not isinstance(down_scale, int):
        'type is:'
        print(type(down_scale))
        raise ValueError('down_scale should be integer. now it is {} with type of '.format(down_scale)+type(down_scale))

    kernelDimension = len(np.shape(input_numpy))
    input_numpy = np.expand_dims(input_numpy[np.newaxis], axis=-1)
    if down_scale == 2:
        kernel_size = 7
    elif down_scale == 4:
        kernel_size = 15
    else:
        raise ValueError('kernel_size is not defined for down_scale={}'.format(str(down_scale)))
    padSize = (np.floor(kernel_size/2)).astype(np.int)
    kenelStrides = tuple([down_scale] * kernelDimension)

    tf.reset_default_graph()
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=np.shape(input_numpy), name="InputImage")
    x_pad = tf.pad(x, ([0, 0], [padSize, padSize], [padSize, padSize], [padSize, padSize], [0, 0]), constant_values=default_pixel_value)
    convKernelGPU = conv_kernel.convDownsampleKernel(kernel_name, kernelDimension, kernel_size, normalizeKernel=normalize_kernel, a=a)
    convKernelGPU = np.expand_dims(convKernelGPU, -1)
    convKernelGPU = np.expand_dims(convKernelGPU, -1)
    convKernelGPU = tf.constant(convKernelGPU)
    y = tf.nn.convolution(x_pad, convKernelGPU, 'VALID', strides=kenelStrides)
    sess.run(tf.global_variables_initializer())
    [output_numpy] = sess.run([y], feed_dict={x: input_numpy})
    if kernelDimension == 2:
        output_numpy = output_numpy[0, :, :, 0]
    if kernelDimension == 3:
        output_numpy = output_numpy[0, :, :, :, 0]

    if mode == 'numpy':
        output = output_numpy
    elif mode == 'sitk':
        output = array_to_sitk(output_numpy, origin=input.GetOrigin(),
                               spacing=tuple(i * down_scale for i in input.GetSpacing()), direction=input.GetDirection())
    return output


# def downsampler_sitk(image_sitk, down_scale, im_ref=None, default_pixel_value=0, interpolator=sitk.sitkBSpline, dimension=3):
#     if im_ref is None:
#         im_ref = sitk.Image(tuple(round(i / down_scale) for i in image_sitk.GetSize()), sitk.sitkInt8)
#         im_ref.SetOrigin(image_sitk.GetOrigin())
#         im_ref.SetDirection(image_sitk.GetDirection())
#         im_ref.SetSpacing(tuple(i * down_scale for i in image_sitk.GetSpacing()))
#     identity = sitk.Transform(dimension, sitk.sitkIdentity)
#     downsampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref, default_pixel_value=default_pixel_value, interpolator=interpolator)
#     return downsampled_sitk


def resampler_sitk(image_sitk, spacing=None, scale=None, im_ref=None, im_ref_size=None, default_pixel_value=0, interpolator=sitk.sitkBSpline, dimension=3):
    """
    :param image_sitk: input image
    :param spacing: desired spacing to set
    :param scale: if greater than 1 means downsampling, less than 1 means upsampling
    :param im_ref: if im_ref available, the spacing will be overwritten by the im_ref.GetSpacing()
    :param im_ref_size: in sikt order: x, y, z
    :param default_pixel_value:
    :param interpolator:
    :param dimension:
    :return:
    """
    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    return resampled_sitk


def SITKshow(img, title=None, margin=0.05, dpi=80):
    import matplotlib.pyplot as plt
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

    t = ax.imshow(nda,
                  extent=extent,
                  interpolation='hamming',
                  cmap='gray',
                  origin='lower')

    if (title):
        plt.title(title)


def index_to_world(landmark_index, spacing=None, origin=None, direction=None, im_ref=None):
    if im_ref is None:
        if spacing is None:
            spacing = [1, 1, 1]
        if origin is None:
            origin = [0, 0, 0]
        if direction is None:
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    else:
        spacing = list(im_ref.GetSpacing())
        origin = list(im_ref.GetOrigin())
        direction = list(im_ref.GetDirection())
    landmarks_point = [None] * len(landmark_index)
    for p in range(len(landmark_index)):
        landmarks_point[p] = [index * spacing[i] + origin[i] for i, index in enumerate(landmark_index[p])]
    return landmarks_point


def world_to_index(landmark_point, spacing=None, origin=None, direction=None, im_ref=None):
    if im_ref is None:
        if spacing is None:
            spacing = [1, 1, 1]
        if origin is None:
            origin = [0, 0, 0]
        if direction is None:
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    else:
        spacing = list(im_ref.GetSpacing())
        origin = list(im_ref.GetOrigin())
        direction = list(im_ref.GetDirection())
    landmarks_index = [None] * len(landmark_point)
    for p in range(len(landmark_point)):
        landmarks_index[p] = [round(point - origin[i] / spacing[i])  for i, point in enumerate(landmark_point[p])]
    return landmarks_index


if __name__ == '__main__':
    input = np.ones((50, 50, 50, 3))
    input_sitk = sitk.GetImageFromArray(input, isVector=1)
    output_sitk = upsampler_gpu(input_sitk, 2, default_pixel_value=0, output_shape_3d=[100, 100, 100])
