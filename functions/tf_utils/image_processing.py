import tensorflow as tf
import numpy as np


def diff(input_tensor, axis=0):
    if axis == 0:
        return input_tensor[1:] - input_tensor[:-1]
    if axis == 1:
        return input_tensor[:, 1:] - input_tensor[:, :-1]
    if axis == 2:
        return input_tensor[:, :, 1:] - input_tensor[:, :, :-1]
    if axis == 3:
        return input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1]
    if axis == 4:
        return input_tensor[:, :, :, :, 1:] - input_tensor[:, :, :, :, :-1]


def bending_energy(dvf, voxel_size=None):
    """
    Bending Energy in TensorFlow:
    :param dvf: with shape of (batch_size, dim0, dim1, dim2, 3)
    :param voxel_size: physical voxel spacing in mm
    :return: 3D bending energy
    """
    if voxel_size is None:
        voxel_size = [1, 1, 1]
    indices_x, indices_y, indices_z = tf.meshgrid(tf.range(0, dvf.get_shape()[1]),
                                                  tf.range(0, dvf.get_shape()[2]),
                                                  tf.range(0, dvf.get_shape()[3]), indexing='ij')
    dvf_tensor = tf.concat([tf.expand_dims(indices_x, -1) * voxel_size[0],
                            tf.expand_dims(indices_y, -1) * voxel_size[1],
                            tf.expand_dims(indices_z, -1)] * voxel_size[2], axis=-1)

    dvf_tensor = tf.expand_dims(dvf_tensor, axis=0)
    dvf_tensor = tf.tile(dvf_tensor, [tf.shape(dvf)[0], 1, 1, 1, 1])
    dvf_tensor = tf.to_float(dvf_tensor)
    dvf_tensor = tf.add(dvf_tensor, dvf)

    dvf_grad_dim0 = diff(dvf_tensor, axis=1) / voxel_size[0]
    dvf_grad_dim1 = diff(dvf_tensor, axis=2) / voxel_size[1]
    dvf_grad_dim2 = diff(dvf_tensor, axis=3) / voxel_size[2]

    dvf_grad_dim0_dim0 = diff(dvf_grad_dim0, axis=1) / voxel_size[0]
    dvf_grad_dim0_dim1 = diff(dvf_grad_dim0, axis=2) / voxel_size[1]
    dvf_grad_dim0_dim2 = diff(dvf_grad_dim0, axis=3) / voxel_size[2]
    dvf_grad_dim1_dim1 = diff(dvf_grad_dim1, axis=2) / voxel_size[1]
    dvf_grad_dim1_dim2 = diff(dvf_grad_dim1, axis=3) / voxel_size[2]
    dvf_grad_dim2_dim2 = diff(dvf_grad_dim2, axis=3) / voxel_size[2]

    dvf_grad_dim0_dim0 = tf.pad(dvf_grad_dim0_dim0, ([0, 0], [0, 2], [0, 0], [0, 0], [0, 0]))
    dvf_grad_dim0_dim1 = tf.pad(dvf_grad_dim0_dim1, ([0, 0], [0, 1], [0, 1], [0, 0], [0, 0]))
    dvf_grad_dim0_dim2 = tf.pad(dvf_grad_dim0_dim2, ([0, 0], [0, 1], [0, 0], [0, 1], [0, 0]))
    dvf_grad_dim1_dim1 = tf.pad(dvf_grad_dim1_dim1, ([0, 0], [0, 0], [0, 2], [0, 0], [0, 0]))
    dvf_grad_dim1_dim2 = tf.pad(dvf_grad_dim1_dim2, ([0, 0], [0, 0], [0, 1], [0, 1], [0, 0]))
    dvf_grad_dim2_dim2 = tf.pad(dvf_grad_dim2_dim2, ([0, 0], [0, 0], [0, 0], [0, 2], [0, 0]))

    smoothness = tf.reduce_mean(tf.square(dvf_grad_dim0_dim0) + 2 * tf.square(dvf_grad_dim0_dim1) + 2 * tf.square(dvf_grad_dim0_dim2) +
                     tf.square(dvf_grad_dim1_dim1) + 2 * tf.square(dvf_grad_dim1_dim2) +
                     tf.square(dvf_grad_dim2_dim2)
                     )
    return smoothness


def calculateSmoothness(DVF, voxelSize=[1, 1]):
    '''
    :param DVF: a numpy array with shape of (2, sizeY, sizeX) or (3, sizeZ, sizeY, sizeX). You might use np.transpose before this function to correct the order of DVF shape.
    :param voxelSize: physical voxel spacing in mm
    :return: Jac
    Hessam Sokooti h.sokooti@gmail.com
    '''

    if (len(np.shape(DVF)) - 1) != len(voxelSize):
        raise ValueError ('dimension of DVF is {} but dimension of voxelSize is {}'.format(
            len(np.shape(DVF)) - 1, len(voxelSize)))
    T = np.zeros(np.shape(DVF), dtype=np.float32) # derivative should be calculated on T which is DVF + indices (world coordinate)
    indices = [None] * (len(np.shape(DVF)) - 1)
    DVF_grad = {}
    if len(voxelSize) == 2:
        indices[0], indices[1] = np.meshgrid(np.arange(0, np.shape(DVF)[1]), np.arange(0, np.shape(DVF)[2]), indexing='ij')
    if len(voxelSize) == 3:
        indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, np.shape(DVF)[1]),
                                                         np.arange(0, np.shape(DVF)[2]),
                                                         np.arange(0, np.shape(DVF)[3]), indexing='ij')

    for d in range(len(voxelSize)):
        indices[d] = indices[d] * voxelSize[d]
        T[d, :] = DVF[d, :] + indices[d]
    for d in range(len(voxelSize)):
        DVF_grad['dim' + str(d)] = []
        DVF_grad['dim'+str(d)] = np.diff(T, n=1, axis=d + 1)
        for d2 in range(d, len(voxelSize)):
            DVF_grad['dim' + str(d), 'dim' + str(d2)] = []
            DVF_grad['dim' + str(d), 'dim' + str(d2)] = np.diff(DVF_grad['dim' + str(d)], n=1, axis=d2 + 1)
            padAfter = np.zeros(len(voxelSize) + 1, dtype=np.int8)      # xyz order
            padAfter[d+1] = padAfter[d+1] + 1
            padAfter[d2+1] = padAfter[d2+1] + 1
            if len(voxelSize) == 2:
                DVF_grad['dim' + str(d), 'dim' + str(d2)] = np.pad(DVF_grad['dim' + str(d), 'dim' + str(d2)],
                                    ((0, 0), (0, padAfter[1]), (0, padAfter[2])), 'constant', constant_values=(0,))
            if len(voxelSize) == 3:
                DVF_grad['dim' + str(d), 'dim' + str(d2)] = np.pad(DVF_grad['dim' + str(d), 'dim' + str(d2)],
                                   ((0, 0), (0, padAfter[1]), (0, padAfter[2]), (0, padAfter[3])), 'constant', constant_values=(0,))

    if len(voxelSize) == 2:
        smoothness = np.mean(np.square(DVF_grad[('dim0', 'dim0')]) + 2 * np.square(DVF_grad[('dim0', 'dim1')]) + np.square(DVF_grad[('dim1', 'dim1')]))
        #       T0/dir0      *   T1/dir1      -    T0/dir1     *   T1/dir0

    if len(voxelSize) == 3:
        smoothness = np.mean(np.square(DVF_grad[('dim0', 'dim0')]) + 2 * np.square(DVF_grad[('dim0', 'dim1')]) + 2 * np.square(DVF_grad[('dim0', 'dim2')]) +
                             np.square(DVF_grad[('dim1', 'dim1')]) + 2 * np.square(DVF_grad[('dim1', 'dim2')]) +
                             np.square(DVF_grad[('dim2', 'dim2')])
                             )
    return smoothness

