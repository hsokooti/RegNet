import copy


def load_deform_exp_setting(selected_deform_exp):
    def_setting = dict()
    if selected_deform_exp == '3D_max20_D14':
        def_setting = dict()
        def_setting['MaxDeform'] = 20  # The maximum amplitude of deformations
        def_setting['DeformMethods'] = ['respiratory_motion', 'respiratory_motion', 'respiratory_motion', 'respiratory_motion',
                                        'single_frequency', 'single_frequency', 'single_frequency', 'single_frequency', 'single_frequency',
                                        'mixed_frequency', 'mixed_frequency', 'mixed_frequency', 'mixed_frequency',
                                        'zero']
        def_setting['UseLungMask'] = True  # The peaks of synthetic deformation can only be inside the mask
        def_setting['verbose_image'] = False  # Detailed writing of images: writing the DVF of the nextFixedImage
        def_setting['DVFNormalization'] = True
        def_setting['MaskToZero'] = 'Torso'
        def_setting['WriteIntermediateIntensityAugmentation'] = False

        # stages
        def_setting['DeleteStage1Images'] = True  # After downsampling, delete all images in the original resolution.

        # images
        def_setting['Canny_LowerThreshold'] = 50.0
        def_setting['Canny_UpperThreshold'] = 100.0
        def_setting['Im_NoiseSigma'] = 10     # Sigma for adding noise after deformation
        def_setting['Im_NoiseAverage'] = 10   # Mean for adding noise after deformation

        # occlusion
        def_setting['Occlusion'] = True
        def_setting['Occlusion_NumberOfEllipse'] = 10
        def_setting['Occlusion_IntensityRange'] = [-800, -780]
        def_setting['Occlusion_Max_a'] = 15
        def_setting['Occlusion_Max_b'] = 15
        def_setting['Occlusion_Max_c'] = 15

        # NextIm
        def_setting['NextIm_SigmaN'] = 2     # The intensity noise is less than normal Defomred Images in order to prevent accumulating noise.
        # Since we are going to generate several deformed images on the NextIm
        def_setting['NextIm_MaxDeform'] = 15

        # Single Frequency
        def_setting['SingleFrequency_BSplineGridSpacing'] = [[80, 80, 80], [70, 70, 70], [60, 60, 60], [50, 50, 50], [45, 45, 45]]  # in mm approximately
        def_setting['SingleFrequency_SetGridBorderToZero'] = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        def_setting['SingleFrequency_GridSmoothingSigma'] = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]  # in voxel not in mm
        def_setting['SingleFrequency_BackgroundSmoothingSigma'] = [8, 8, 8, 8, 8]  # in voxel not in mm
        def_setting['SingleFrequency_MaxDeformRatio'] = [1, 1, 1, 1, 1]

        # Mixed Frequency
        def_setting['MixedFrequency_BlockRadius'] = 20  # in voxel not in mm
        def_setting['MixedFrequency_Np'] = [200, 150, 150, 150]
        def_setting['MixedFrequency_BSplineGridSpacing'] = [[80, 80, 80], [60, 60, 60], [50, 50, 50], [45, 45, 60]]
        def_setting['MixedFrequency_SigmaRange'] = [[10, 15], [10, 15], [10, 15], [10, 15]]  # in voxel not in mm
        def_setting['MixedFrequency_GridSmoothingSigma'] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # in voxel not in mm
        def_setting['MixedFrequency_SetGridBorderToZero'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # in voxel not in mm
        def_setting['MixedFrequency_MaxDeformRatio'] = [1, 1, 1, 1]

        # Respiratory Motion
        def_setting['RespiratoryMotion_t0'] = [30, 30, 30, 30, 30]  # in mm
        def_setting['RespiratoryMotion_s0'] = [0.12, 0.12, 0.12, 0.12, 0.12]
        def_setting['RespiratoryMotion_BSplineGridSpacing'] = [[80, 80, 80], [70, 70, 70], [60, 60, 60], [50, 50, 50], [45, 45, 45]]  # in mm approximately
        def_setting['RespiratoryMotion_SetGridBorderToZero'] = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        def_setting['RespiratoryMotion_GridSmoothingSigma'] = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]  # in voxel not in mm
        def_setting['RespiratoryMotion_BackgroundSmoothingSigma'] = [8, 8, 8, 8, 8]  # in voxel not in mm
        def_setting['RespiratoryMotion_MaxDeformRatio'] = [1, 1, 1, 1, 1]
        def_setting['RespiratoryMotion_SingleFrequency_MaxDeformRatio'] = [0.5, 0.5, 0.5, 0.5, 0.5]

        # translation
        def_setting['Translation_MaxDeformRatio'] = [1, 1, 1, 1]

        # translation_old
        def_setting['BsplineGridSpacing_translation'] = [[40, 40, 40], [40, 40, 40], [40, 40, 40], [40, 40, 40]]
        def_setting['setGridBorderToZero_translation'] = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]

    elif selected_deform_exp == '3D_max20_D14_K':
        deform_exp_setting_temp = load_deform_exp_setting('3D_max20_D14')
        def_setting = copy.deepcopy(deform_exp_setting_temp)

    elif selected_deform_exp in ['3D_max7_D14_K']:
        deform_exp_setting_temp = load_deform_exp_setting('3D_max20_D14')
        def_setting = copy.deepcopy(deform_exp_setting_temp)
        def_setting['MaxDeform'] = 7
        def_setting['SingleFrequency_BSplineGridSpacing'] = [[50, 50, 50], [45, 45, 45], [35, 35, 35], [25, 25, 25], [20, 20, 20]]
        def_setting['SingleFrequency_MaxDeformRatio'] = [0.5, 1, 1, 1, 1]
        def_setting['MixedFrequency_BSplineGridSpacing'] = [[50, 50, 50], [40, 40, 40], [25, 25, 35], [20, 20, 30]]
        def_setting['MixedFrequency_SigmaRange'] = [[5, 10], [5, 10], [5, 10], [5, 10]]
        def_setting['MixedFrequency_MaxDeformRatio'] = [1, 1, 1, 1]
        def_setting['RespiratoryMotion_t0'] = [15, 15, 15, 15, 15]  # in mm
        def_setting['RespiratoryMotion_s0'] = [0.12, 0.12, 0.12, 0.12, 0.12]
        def_setting['RespiratoryMotion_BSplineGridSpacing'] = [[50, 50, 50], [45, 45, 45], [35, 35, 35], [25, 25, 25], [20, 20, 20]]
        def_setting['RespiratoryMotion_MaxDeformRatio'] = [1, 1, 1, 1, 1]
        def_setting['RespiratoryMotion_SingleFrequency_MaxDeformRatio'] = [0.5, 0.5, 0.5, 0.5, 0.5]

    elif selected_deform_exp in ['3D_max15_D14_K']:
        deform_exp_setting_temp = load_deform_exp_setting('3D_max20_D14')
        def_setting = copy.deepcopy(deform_exp_setting_temp)
        def_setting['MaxDeform'] = 15
        def_setting['SingleFrequency_BSplineGridSpacing'] = [[60, 60, 60], [50, 50, 50], [45, 45, 45], [40, 40, 40], [35, 35, 35]]
        def_setting['SingleFrequency_MaxDeformRatio'] = [0.5, 1, 1, 1, 1]
        def_setting['MixedFrequency_BSplineGridSpacing'] = [[60, 60, 60], [50, 50, 40], [40, 40, 80], [35, 35, 80]]
        def_setting['MixedFrequency_SigmaRange'] = [[7, 12], [7, 12], [7, 12], [7, 12]]
        def_setting['MixedFrequency_MaxDeformRatio'] = [1, 1, 1, 1]
        def_setting['RespiratoryMotion_t0'] = [22, 22, 22, 22, 22]  # in mm
        def_setting['RespiratoryMotion_s0'] = [0.12, 0.12, 0.12, 0.12, 0.12]
        def_setting['RespiratoryMotion_BSplineGridSpacing'] = [[60, 60, 60], [50, 50, 50], [45, 45, 45], [40, 40, 40], [35, 35, 35]]
        def_setting['RespiratoryMotion_MaxDeformRatio'] = [1, 1, 1, 1, 1]
        def_setting['RespiratoryMotion_SingleFrequency_MaxDeformRatio'] = [0.5, 0.5, 0.5, 0.5, 0.5]

    else:
        print('warning: -------- selected_deform_exp not found')
    return def_setting
