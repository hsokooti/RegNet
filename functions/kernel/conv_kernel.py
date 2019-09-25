import numpy as np


def linear1d(x):
    x = abs(x)
    if 0 >= x < 1:
        f = -x + 1    
    else:
        f = 0
    return f


def linear2d(x, y):
    x = abs(x)
    y = abs(y)    
    if 0 >= (x+y) < 1:
        f = -x-y + 1    
    else:
        f = 0
    return f


def linear3d(x, y, z):
    x = abs(x)
    y = abs(y)    
    z = abs(z)
    if (((x+y+z)>= 0) and ((x+y+z) <1) ):
        f = -x-y-z + 1    
    else:
        f = 0
    return f


def spline1D(x, a= -0.5):    
    x = abs(x)
    if (x>=0 and x<1):
        f = (a+2)*(x**3) - (a+3)*(x**2) + 1
    elif (x>=1 and x<=2):
        f = (a)*(x**3) - 5*(a)*(x**2) + 8*a*x - 4*a        
    else: 
        f = 0
    return f


def spline2D(x, y, a= -0.5):    
    x = abs(x)
    y = abs(y)
    if ( ((x+y)>=0) and ((x+y)<1)):
        f = (a+2)*((x+y)**3) - (a+3)*((x+y)**2) + 1
    elif ((x+y)>=1 and (x+y)<=2):
        f = (a)*((x+y)**3) - 5*(a)*((x+y)**2) + 8*a*(x+y) - 4*a        
    else: 
        f = 0
    return f


def spline3D(x, y, z, a= -0.5):    
    x = abs(x)
    y = abs(y)
    z = abs(z)
    if ( ((x+y+z)>=0) and ((x+y+z)<1)):
        f = (a+2)*((x+y+z)**3) - (a+3)*((x+y+z)**2) + 1
    elif ((x+y+z)>=1 and (x+y+z)<=2):
        f = (a)*((x+y+z)**3) - 5*(a)*((x+y+z)**2) + 8*a*(x+y+z) - 4*a        
    else: 
        f = 0
    return f


def bspline1D(x):
    x = abs(x)
    if ( x>=0 and x<1):
        f = 0.5*(x**3) - x**2 + 4/6
    elif (x>=1 and x<=2):
        f = (-1/6)*(x**3) + x**2 -2*x + 8/6
    else:
        f = 0
    return f


def bspline2D(x, y):
    x = abs(x)
    y = abs(y)
    if ( ((x+y)>=0) and ((x+y)<1)):
        f = (0.5)*((x+y)**3) - ((x+y)**2) + 4/6
    elif ((x+y)>=1 and (x+y)<=2):
        f = (-1/6)*((x+y)**3) + ((x+y)**2) -2*(x+y) + 8/6
    else:
        f = 0
    return f

def bspline3D(x, y, z):
    x = abs(x)
    y = abs(y)
    z = abs(z)
    if ( ((x+y+z)>=0) and ((x+y+z)<1)):
        f = (0.5)*((x+y+z)**3) - ((x+y+z)**2) + 4/6
    elif ((x+y+z)>=1 and (x+y+z)<=2):
        f = (-1/6)*((x+y+z)**3) + ((x+y+z)**2) -2*(x+y+z) + 8/6
    else:
        f = 0
    return f


def mitchell1D(x, b=1/3, c=1/3):
    x = abs(x)
    if (x >= 0 and x < 1):
        f = (1/6)*((12 - 9*b - 6*c)*(x**3) + (-18 + 12*b + 6*c)*(x**2) + 6 - 2*b)
    elif (x >= 1 and x <= 2):
        f = (1/6)*((-b - 6*c)*(x**3) + (6*b + 30*c)*(x**2) + (-12*b - 48*c)*x + 8*b + 24*c)
    else:
        f = 0
    return f


def convDownsampleKernel(kernelName, dimension, kernelSize , a=-0.5, b=1/3, c=1/3, normalizeKernel=None):
    numOfPoints = kernelSize + 2
    XInput = np.linspace(-2, 2, num=numOfPoints )
    # print ('distance between samples is {:0.2f} voxel. This is suitable for downsampling with the factor of {:0.2f} '
    #        .format(XInput[1]-XInput[0], 1/(XInput[1]-XInput[0])))
    if dimension == 1:            
        if kernelName == 'linear':
            Y = np.stack([linear1d (XInput[i]) for i in range(0, len(XInput))])
        elif kernelName == 'spline':
            Y = np.stack([spline1D (XInput[i], a = a) for i in range(0, len(XInput))])
        elif kernelName == 'bspline':
            Y = np.stack([bspline1D (XInput[i]) for i in range(0, len(XInput))])
        elif kernelName == 'mitchell':
            Y = np.stack([bspline1D (XInput[i]) for i in range(0, len(XInput))])
        else:
            raise ValueError('cannot find the  kernel ' + kernelName + ' !')
        Y = Y[1:-1]

    if dimension == 2:
        YInput = np.linspace(-2, 2, num=numOfPoints )
        xv, yv = np.meshgrid(XInput, YInput)
        if kernelName == 'linear':
            Y = np.stack([linear2d(xv[i, j], yv[i, j]) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])])
        if kernelName == 'spline':
            Y = np.stack([spline2D(xv[i,j],yv[i,j],  a=a) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])])
        if kernelName == 'bspline':
            Y = np.stack([bspline2D(xv[i,j],yv[i,j]) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])])
        Y = np.reshape(Y , [len(XInput),len(XInput)])
        Y = Y[1:-1, 1:-1]

    if dimension == 3:
        YInput = np.linspace(-2, 2, num=numOfPoints)
        ZInput = np.linspace(-2, 2, num=numOfPoints)
        xv, yv, zv = np.meshgrid(XInput, YInput, ZInput)             
        if kernelName == 'linear':
            Y = np.stack([linear3d(xv[i, j, k], yv[i, j, k], zv[i, j, k]) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])
                          for k in range(0, np.shape(xv)[0])])
        if kernelName == 'spline':
            Y = np.stack([spline3D(xv[i,j,k], yv[i,j,k], zv[i,j,k], a=a) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])
            for k in range(0, np.shape(xv)[0])])
        if kernelName == 'bspline':
            Y = np.stack([bspline3D(xv[i,j,k], yv[i,j,k], zv[i,j,k]) for i in range(0, np.shape(xv)[0]) for j in range(0, np.shape(xv)[0])
            for k in range(0, np.shape(xv)[0])])
        Y = np.reshape(Y , [len(XInput),len(XInput), len(XInput)])
        Y = Y[1:-1, 1:-1, 1:-1]
    if normalizeKernel:
        if np.sum(Y) != normalizeKernel:
            ratio = normalizeKernel / np.sum(Y)
            Y = ratio * Y

    Y[abs(Y) < 1e-6] = 0
    return Y.astype(np.float32)


def bilinear_up_kernel(dim=3, kernel_size=3):
    """
    bilinear kernel for upsampling with transposed convolution.

    :param dim: 1, 2, 3
    :param kernel_size: 3 or 5
    :return:
    """
    if kernel_size not in [3, 5]:
        raise ValueError('kernel_size should be in [3, 5]')

    center = kernel_size // 2
    if dim == 1:
        indices = np.arange(0,3)
        indices = indices - center
        distance_to_center = np.absolute(indices)
        kernel = (np.ones(np.shape(indices)) / (np.power(2, distance_to_center))).astype(np.float32)

    elif dim == 2:
        indices = [None] * dim
        indices[0], indices[1], = np.meshgrid(np.arange(0, 3), np.arange(0, 3), indexing='ij')
        for i in range(0, dim):
            indices[i] = indices[i] - center
        distance_to_center = np.absolute(indices[0]) + np.absolute(indices[1])
        kernel = (np.ones(np.shape(indices[0])) / (np.power(2, distance_to_center))).astype(np.float32)

    elif dim == 3:
        indices = [None] * dim
        indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, 3), np.arange(0, 3), np.arange(0, 3), indexing='ij')
        for i in range(0, dim):
            indices[i] = indices[i] - center
        distance_to_center = np.absolute(indices[0]) + np.absolute(indices[1]) + np.absolute(indices[2])
        kernel = (np.ones(np.shape(indices[0])) / (np.power(2, distance_to_center))).astype(np.float32)

    else:
        raise ValueError('bilinear_up_kernel is not defined for dimension larger than 3')
    return kernel
