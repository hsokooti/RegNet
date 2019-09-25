

RegNet
==========

## Introduction
In this work we propose a method to solve nonrigid image registration through a learning approach, instead of via iterative optimization of a predefined dissimilarity metric. We design a Convolutional Neural Network (CNN) architecture that, in contrast to all other work, directly estimates the displacement vector field (DVF) from a pair of input images. The proposed RegNet is trained using a large set of artificially generated DVFs, does not explicitly define a dissimilarity metric, and integrates image content at multiple scales to equip the network with contextual information. At testing time nonrigid registration is performed in a single shot, in contrast to current iterative methods.

### Citation
[1] Sokooti, H., de Vos, B., Berendsen, F., Ghafoorian, M., Yousefi, S., Lelieveldt, B.P., Isgum, I. and Staring, M., 2019. 3D Convolutional Neural Networks Image Registration Based on Efficient Supervised Learning from Artificial Deformations. arXiv preprint arXiv:1908.10235.

[2] Sokooti, H., de Vos, B., Berendsen, F., Lelieveldt, B.P., Išgum, I. and Staring, M., 2017, September. Nonrigid image registration using multi-scale 3D convolutional neural networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 232-239). Springer, Cham.

	
## 1. Dependencies
- [Joblib](http://github.com/joblib/joblib) : Running Python functions as pipeline jobs.
- [Matplotlib](https://matplotlib.org/) A plotting library for the Python programming language and its numerical mathematics extension NumPy.
- [NumPy](http://www.numpy.org/) : General purpose array-processing package.
- [SimpleITK](http://www.simpleitk.org/) : Simplified interface to the Insight Toolkit for image registration and segmentation.
- [SciPy](https://www.scipy.org/) : A Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [TensorFlow v1.x](https://www.tensorflow.org/) : TensorFlow helps the tensors flow.
- [xmltodict](https://github.com/martinblech/xmltodict) : Python module that makes working with XML feel like you are working with JSON.
	

## 2. Running RegNet
Run`RegNet3D.py`. Please note that current RegNet only works with 3D images.

### 2.1 Data
All images are read and written by [SimpleITK](http://www.simpleitk.org/). The images are already resampled to an isotropic voxel size of [1, 1, 1] mm.

The images in the training and validation set can be defined in a list of dictionaries: 
```python
# simple example how to load the data:

import functions.setting.setting_utils as su


setting = su.initialize_setting(current_experiment='MyCurrentExperiment', where_to_run='Root')
data_exp_dict = [{'data': 'SPREAD',                              # Data to load. The image addresses can be modified in setting_utils.py
		  'deform_exp': '3D_max7_D14_K',                    # Synthetic deformation experiment
		  'TrainingCNList': [i for i in range(1, 11)],   # Case number of images to load (The patient number)
		  'TrainingTypeImList': [0, 1],                  # Types images for each case number, for example [baseline, follow-up]
		  'TrainingDSmoothList': [i for i in range(9)],  # The synthetic type to load. For instance, ['translation', 'bsplineSmooth']
		  'ValidationCNList': [11, 12],
		  'ValidationTypeImList': [0, 1],
		  'ValidationDSmoothList': [2, 4, 8],
		  },
		 {'data': 'DIR-Lab_4D',
		  'deform_exp': '3D_max7_D14_K',
		  'TrainingCNList': [1, 2, 3],
		  'TrainingTypeImList': [i for i in range(8)],
		  'TrainingDSmoothList': [i for i in range(9)],
		  'ValidationCNList': [1, 2],
		  'ValidationTypeImList': [8, 9],
		  'ValidationDSmoothList': [2, 4, 8],
		  }
		 ]

setting = su.load_setting_from_data_dict(setting, data_exp_dict)
original_image_address = su.address_generator(setting, 'OriginalIm', data='DIR-Lab_4D', cn=1, type_im=0, stage=1)
print(original_image_address)

```
`./Data/DIR-Lab/4DCT/mha/case1/case1_T00_RS1.mha`

#### `'data'`: 
The details of `'data'` should be written in the `setting_utils.py`. The general setting of each `'data'` should be defined in 
`load_data_setting(selected_data)` like the extension, total number of types and default pixel value. The global data folder (`setting['DataFolder']`) can be defined in `root_address_generator(where_to_run='Auto')`. 

The details of the image address can be defined in `address_generator()` after the line `if data == 'YourOwnData':`. For example you can take a look at the line 370: `if data == 'DIR-Lab_4D':`. The orginal images are defined with `requested_address= 'originalIm'`. To test the reading function, you can run the above script and check the `original_image_address`.


#### `'deform_exp', 'TrainingDSmoothList'`: 
check section 2.2.4 Setting of generating synthetic DVFs

#### `'TrainingCNList', 'TrainingTypeImList'`: 
`'TrainingCNList'` indicates the Case Numbers (CN) that you want to use for training. Usually each cn refers to a specific patient. `'TrainingTypeImList'` indicates which types of the available images for each patient you want to load. For example in the SPREAD data, two types are available: baseline and follow-up. In the DIR-Lab_4D data, for each patient 10 images are available from the maximum inhale to maximum exhale phase.

### 2.2 Setting of generating synthetic DVFs
Three categories of synthetic DVF are available in the software: translation, single frequency, mixed frequency
#### 2.2.1 Zero Frequency `'translation'`
#### 2.2.2 Single frequency `'smoothBspline'`
For generating single-frequency DVF, we proposed the following algorithm:
1. Initialize a B-spline grid points with a grid spacing of `deform_exp_setting['BsplineGridSpacing_smooth']`.
2. Perturb the gird points in a smooth and random fashion.
3. Interpolate to get the DVF.
4. Normalize the DVF linearly, if it is out of the range `[-deform_exp_setting['MaxDeform'], +deform_exp_setting['MaxDeform']]`.
By varying the spacing, different spatial frequencies are generated.
![alt text](Documentation/SyntheticDVF_SingleFreq.png "Single Frequency")
<p align="center">Figure 1: Single Frequency: B-spline grid spacing are 40, 30 and 20 mm from left to right.</p>

#### 2.2.3 Mixed frequency `'dilatedEdge'`

The steps for the mixed-frequency category is as follows:
1. Extract edges with Canny edge detection method.
2. Copy the binary image three times to get a vector of 3D image with the length of three.
3. Set some voxels to be zero randomly for each image. 
4. Dilate the binary image for `deform_exp_setting['Np_dilateEdge']` iteration by using a random structure element for each image.
5. Fill the binary dilated image with a DVF generated from the single-frequency method.
6. Smooth the DVF with a Gaussian kernel with standard deviation of `deform_exp_setting['sigmaRange_dilatedEdge']`. The sigma is relatively small which leads to a higher spatial frequency in comparison with the filled DVF.
By varying the sigma value and `deform_exp_setting['BsplineGridSpacing_dilatedEdge']` in the filled DVF, different spatial frequencies will be mixed together.

![alt text](Documentation/SyntheticDVF_MixedFreq.png "Mixed Frequency")
<p align="center">Figure 2: Mixed Frequency.</p>

#### 2.2.4 `'deform_exp', 'TrainingDSmoothList'`
`'deform_exp'` is defined in the `setting_utils.py` with the function `load_deform_exp_setting(selected_deform_exp)`. For example you can use three types of translation, single frequency and mixed frequency:
```python
deform_exp_setting['deformMethods'] = ['translation', 'translation', 'translation',
				       'smoothBSpline', 'smoothBSpline', 'smoothBSpline',
				       'dilatedEdgeSmooth', 'dilatedEdgeSmooth', 'dilatedEdgeSmooth']
```
The above setting is at the generation time. However, you might not want to load all of them at the reading time:

`'ValidationDSmoothList': [2, 4, 8]`: This means that you want to load translation type2, smoothBspline type1 and dilatedEdgeSmooth type 2.

### 2.3 Network
The proposed network is given in Figure 3.
![alt text](Documentation/RegNet2.PNG "RegNet design")
<p align="center">Figure 3: RegNet design.</p>

### 2.4 Software Architecture
![alt text](Documentation/Software_Architecture2.PNG "Software Architecture")
<p align="center">Figure 4: Software Architecture.</p>

#### 2.4.1 Memory efficiency
It is not efficient (or possible)  to load all images with their DVFs to the memory. A DVF is three times bigger than its corresponding image with the type of float32. Alternatively, this software loads a chunk of images.  The number of images per chunk can be chosen by the parameter: `setting['NetworkTraining']['NumberOfImagesPerChunk']`
```python
setting['NetworkTraining']['NumberOfImagesPerChunk'] = 16  # Number of images that I would like to load in RAM
setting['NetworkTraining']['SamplesPerImage'] = 50
setting['NetworkTraining']['BatchSize'] = 15
setting['NetworkTraining']['MaxQueueSize'] = 20
```

#### 2.4.2 Parallel Computing
We used `threading` in order to read patches in parallel with training the network. We define the `functions.reading.direct` class to read in a normal way and the `functions.reading.thread` class to read patches with threading.


