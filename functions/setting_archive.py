
setting = {}

if '3D_max20' in setting['DeformExpList']:
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. , noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [20, 20, 20]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = False  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = False  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)
    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

if '3D_max20_Np10' in setting['DeformExpList']:
    setting['Dim'] = '3D'  # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [20, 20, 20]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [10, 10, 10]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = False  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = False  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 5

if '3D_max15' in setting['DeformExpList']:
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [15, 15, 15]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = False  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = False  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

if setting['deformName'] == '3D_max15_E':
    setting['Dim'] = '3D'  # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [15, 15, 15]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)
    setting['onEdge-lowerThreshold'] = 50.0
    setting['onEdge-upperThreshold'] = 100.0

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

elif setting['deformName'] == '3D_max10':
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [10, 10, 10]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['voxelSize'] = [1, 1, 1]
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = False  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = False  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

elif setting['deformName'] == '3D_max10_E':
    setting['Dim'] = '3D'  # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [10, 10, 10]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)
    setting['onEdge-lowerThreshold'] = 50.0
    setting['onEdge-upperThreshold'] = 100.0

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

elif setting['deformName'] == '3D_max15_ED':
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [10, 10, 10]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)
    setting['onEdge-lowerThreshold'] = 50.0
    setting['onEdge-upperThreshold'] = 100.0

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

    setting['blockRadius_dilatedEdge'] = 20
    setting['MaxDeform_dilateEdge'] = [50, 50, 50]
    setting['Np_dilateEdge'] = 100
    setting['BsplineGridSpacing_dilatedEdge'] = [[80, 80, 80], [80, 80, 80], [80, 80, 80]]
    setting['sigmaRange_dilatedEdge'] = [[5, 9], [5, 9], [5, 9]]
    setting['WriteDVFStatistics'] = False

elif setting['deformName'] == '3D_max15_ED':
    setting['Dim'] = '3D'  # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    setting['DistanceDeform'] = 40  # The minimum distance between two random peaks
    setting['DistanceArea'] = 20  # The area that is inculeded in the training algorithm
    setting['sigmaNL'] = 1  # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    setting['Border'] = 33  # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    setting['sigmaN'] = 5  # Sigma for adding noise after deformation
    setting['deformMethod'] = [0, 1, 2]  # 0: low freq, 1: medium freq, 2: high freq.
    setting['MaxDeform'] = [15, 15, 15]  # The maximum amplitude of deformations
    setting['sigmaB'] = [35, 25, 20]  # For blurring deformaion peak
    setting['Np'] = [100, 100, 100]  # Number of random peaks
    setting['DVFNormalization'] = True
    setting['DSmoothMethod'] = [0, 1, 2, 3, 4, 5]  # 0: no nextIm, 1: one nextIm, 2: two nextIm.
    setting['defaultPixelValue'] = -2048  # The pixel value when a transformed pixel is outside of the image
    setting['voxelSize'] = [1, 1, 1]
    setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
    setting['loadMask'] = True  # The peaks of synthetic deformation can only be inside the mask
    setting['onEdge'] = True  # The peaks of synthetic deformation can only be on edges (calculated by sitk.CannyEdgeDetection)
    setting['onEdge-lowerThreshold'] = 50.0
    setting['onEdge-upperThreshold'] = 100.0

    setting['Border_nextIm'] = 33
    setting['sigmaN_nextIm'] = 2  # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise. Since we are going to generate several deformed images on the nextIm
    setting['MaxDeform_nextIm'] = 15
    setting['sigmaB_nextIm'] = 35  # Low frequency deformation is chosen for the nextIm. We just need a slightly deformed image
    setting['Np_nextIm'] = 100

    setting['blockRadius_dilatedEdge'] = 20
    setting['MaxDeform_dilateEdge'] = [50, 50, 50]
    setting['Np_dilateEdge'] = 100
    setting['BsplineGridSpacing_dilatedEdge'] = [[80, 80, 80], [80, 80, 80], [80, 80, 80]]
    setting['sigmaRange_dilatedEdge'] = [[5, 9], [5, 9], [5, 9]]
    setting['WriteDVFStatistics'] = False