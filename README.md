

RegNet
==========

## Introduction
In this paper we propose a method to solve nonrigid image registration through a learning approach, instead of via iterative optimization of a predefined dissimilarity metric. We design a Convolutional Neural Network (CNN) architecture that, in contrast to all other work, directly estimates the displacement vector field (DVF) from a pair of input images. The proposed RegNet is trained using a large set of artificially generated DVFs, does not explicitly define a dissimilarity metric, and integrates image content at multiple scales to equip the network with contextual information. At testing time nonrigid registration is performed in a single shot, in contrast to current iterative methods.

### Citation

[1] Sokooti, H., de Vos, B., Berendsen, F., Lelieveldt, B.P., Išgum, I. and Staring, M., 2017, September. Nonrigid image registration using multi-scale 3D convolutional neural networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 232-239). Springer, Cham.

	
## 1. Dependencies
- [TensorFlow](https://www.tensorflow.org/) : TensorFlow helps the tensors flow.
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
- [SciPy](https://www.scipy.org/) : A Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [SimpleITK](http://www.simpleitk.org/) : Simplified interface to the Insight Toolkit for image registration and segmentation.
	

## 2. Running RegNet
Run either `RegNet2D_MICCAI.py` or `RegNet3D_MICCAI.py`. Please note that current RegNet only works with 3D images and the script `RegNet2D_MICCAI.py` extracts 2D slices from a 3D image.

### 2.1 Data
Images are read and written by [SimpleITK](http://www.simpleitk.org/).  Check the documentation for the image type. Images are already resampled to an isotropic voxel size of [1, 1, 1] mm.

This software considers both fixed and moving images are available in the database. It reads images with the following addresses: 
```
Setting['DLFolder']/Elastix/LungExp/ExpLung1/Result/FixedImageFullRS1.mha
Setting['DLFolder']/Elastix/LungExp/ExpLung1/Result/MovingImageFullRS1.mha
Setting['DLFolder']/Elastix/LungExp/ExpLung2/Result/FixedImageFullRS1.mha
...
```


### 2.2 Network
The proposed network is given in Figure 1.
![alt text](Documentation/RegNet.PNG "RegNet design")
<p align="center">Figure 1: RegNet design.</p>

### 2.3 Setting of generating synthetic DVFs

Synthetic DVFs are generated with varying spatial frequency and amplitude, aiming to represent the range of displacements that can be seen in real images. The parameter `Setting['sigmaB']` controls the spatial frequencies of the synthetic DVFs. (See Figure 2)

    Setting['DLFolder'] = '/hsokooti/DL/'    
    Setting['deformName'] = 'LungExp2D_1'
    Setting['Dim'] = '2D'               # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    Setting['DistanceDeform'] = 40      # The minimum distance between two random peaks
    Setting['DistanceArea'] = 20        # The area that is inculeded in the training algorithm
    Setting['sigmaNL'] = 1              # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    Setting['Border'] = 33              # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    Setting['sigmaN'] = 5               # Sigma for adding noise after deformation
    Setting['MaxDeform'] = [20, 15, 15] # The maximum amplitude of deformations
    Setting['sigmaB'] = [35, 25, 20]    # For blurring deformaion peak
    Setting['Np'] = [100, 100, 100]     # Number of random peaks

The above setting makes the following deformed images: (3 deformed images of the fixed image and 3 deformed images of the moving image)
```
Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D0/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D0/DeformedDVF.mha

Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D1/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D1/DeformedDVF.mha

Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D2/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1F/ExpLung1/Dsmooth0/D2/DeformedDVF.mha

Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D0/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D0/DeformedDVF.mha

Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D1/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D1/DeformedDVF.mha

Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D2/DeformedImage.mha
Setting['DLFolder']/LungExp2D_1M/ExpLung1/Dsmooth0/D2/DeformedDVF.mha
...
```


![alt text](Documentation/syntheticDVF.PNG "syntheticDVF")
<p align="center">Figure 2: Heat maps of the magnitude of DVFs used for training RegNet (left) Low frequency, (middle) Medium frequency, (right) High frequency.</p>

### 2.4 Setting of reading synthetic DVFs

The software extracts patches from images and their DVF in a random fashion. However, it is possible to control it with some settings.  By the use of `Setting['classBalanced']`, training data can be balanced with respect to the values of DVF and at the same time the maximum value of DVFs  is set.

    Setting['Resolution'] = 'multi'         # 'single' or 'multi' resolution. In multiresolution, the downsampled patch is involved.
    Setting['deformMethod'] = [0, 1, 2]     # 0: low freq, 1: medium freq, 2: high freq.
    Setting['classBalanced'] = [1.5, 4, 8]  # Use these threshold values to balance the number of data in each category. for instance [a,b] implies classes [0,a), [a,b). Numbers are in mm
    Setting['K'] = 65                       # Margin from the border to select random patches
    Setting['ParallelSearch'] = True        # Using np.where in parallel with [number of cores - 2] in order to make balanced data. This is done with joblib library
    Setting['R'] = 14                       # Radius of normal resolution patch size. Total size is (2*R +1)
    Setting['Rlow'] = 26                    # Radius of low resolution patch size. Total size is (Rlow +1). Selected patch size: center-Rlow : center+Rlow : 2
    Setting['Ry'] = 0                       # Radius of output. Total size is (2*Ry +1)

#### 2.4.1 Memory efficiency
It is not efficient (/possible)  to load all images with their DVFs to the memory. A DVF is three times bigger than its corresponding image with type of float32. Alternatively, this software loads a chunk of images.  The number of images per chunk can be chosen by the parameter: `numberOfImagesPerChunk`
```
numberOfImagesPerChunk = 5          # Number of images that I would like to load in RAM
samplesPerImage = 10000
```

#### 2.4.2 Threading
We used `threading` in order to read patches in parallel with training the network. We define the `RegNet.Patches` class to read in a normal way and the `RegNetThread.PatchesThread` class to read patches with threading.

#### 2.4.3 Algorithm







