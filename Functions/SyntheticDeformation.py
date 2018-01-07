from __future__ import print_function, division
import os, time, sys, os.path
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import Functions.PyFunctions as PF
import numpy as np
import matplotlib.pyplot as plt

# %%-------------------------------------------h.sokooti@gmail.com--------------------------------------------

class SyntheticDVF(object):
    def __init__(self,
                setting ={},
                ImageType = 0 ,      # 0: Fixed image, 1: Moving image
                IN = 1,              # number of the image in the database. In SPREAD database it can be between 1 and 21. (Please note that it starts from 1 not 0)
                DeformName = '',     # Name of the folder to write or to read.
                Dsmooth = 0,         # This variable is used to generate another deformed version of the moving image. Then, use that image to make synthetic DVFs. More information available on [sokooti2017nonrigid]
                D = 0,               # 0: low freq, 1: medium freq, 2: high freq. More information available on [sokooti2017nonrigid]
                Ini = 0
               ):
        self._setting = setting
        self._ImageType = ImageType
        self._IN = IN
        self._DeformName = DeformName
        self._Dsmooth = Dsmooth
        self._D = D
        self._Ginfo = None
        self._FixedIm = None
        self._FixedIm_ = None
        self._DefomedIm_ = None
        self._DeformedDVF_ = None
        self._Ini = Ini


    def GetDVFandDeformedImages (self):

        '''
        This function generates the synthetic displacement vector fields and writes them to disk and returns them.
        If synthetic DVFs are already generated and can be found on the disk, this function just reads and returns them.
        Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction.

        :return FixedIm_        Fixed image with mentioned ImageType and IN
        :return DefomedIm_      Deformed image by applying the synthetic DeformedDVF_ on the FixedIm_
        :return DeformedDVF_    Syntethic DVF

        Hessam Sokooti h.sokooti@gmail.com
        '''
        Dsmooth = self._Dsmooth
        D = self._D
        # %%----------------------------------------Initial Parameters--------------------------------------------
        DLFolder = self._setting['DLFolder']
        self._Ginfo=PF.MakeGinfo(DLFolder)
        RootPathDL=self._Ginfo['RootPathDL']

        # %%------------------------------------Deformation Parameters--------------------------------------------
        DeformTotNames=[self._DeformName+'F',self._DeformName+'M']
        typeTot=['Fixed','Moving']
        DeformFolder=DeformTotNames[self._ImageType]
        DeformPath=self._setting['DLFolder']+'Elastix/'+DeformFolder+'/'
        typeExp =typeTot[self._ImageType]
        ExpN = self._Ginfo['ExpN'] + str(self._IN)

        if (Dsmooth==0):
            self._FixedIm= sitk.ReadImage(RootPathDL+ExpN+'/Result/'+typeExp+'ImageFullRS1.mha')
    #                FixedMask=sitk.ReadImage(RootPathDL+ExpN+'/Result/'+typeExp+'MaskFullRS1.mha')
        else:
            self._FixedIm = sitk.ReadImage(DeformPath+ExpN+'/Dsmooth'+str(0)+'/DIni'+str(Dsmooth)+'/FixedImageNext.mha')
    #                FixedMask = sitk.ReadImage(DeformPath+ExpN+'/Dsmooth'+str(0)+'/DIni'+str(Dsmooth)+'/FixedMaskNext.mha')
        self._FixedIm_=sitk.GetArrayFromImage(self._FixedIm)
    #            FixedImCanny=sitk.CannyEdgeDetection(FixedIm,50,100)
        # sitk.WriteImage(sitk.Cast(Structure,sitk.sitkInt8),DeformPath+ExpN+'/Dsmooth'+str(Dsmooth)+'/Canny.mha')
        start_time = time.time()
        DeforemdDVFAddress=DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/' + 'DeformedDVF.mha'
        DeforemdImAddress=DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/' + 'DeformedImage.mha'

        if os.path.isfile(DeforemdDVFAddress):
            # If DVF is already generated for this ImageType and IN, then no need to generate them again. We just read them
            DeformedDVF = sitk.ReadImage(DeforemdDVFAddress)
            self._DeformedDVF_=sitk.GetArrayFromImage(DeformedDVF)
            DefomedIm = sitk.ReadImage(DeforemdImAddress)
            self._DefomedIm_=sitk.GetArrayFromImage(DefomedIm)
        else:
            self._DeformedDVF_ = np.zeros([self._FixedIm_.shape[0], self._FixedIm_.shape[1], self._FixedIm_.shape[2], 3], dtype=np.float64)
            if D<2:
                # smooth and blob are very similar. The only difference is that in blob3 we try to avoid choosing peak point in the neighborhood of the previous points.
                # the BorderMask_ and IEdge would change in each iteration to find the points.
                start_time = time.time()
                self.smooth()
            if D==2:
                self.blob()

            DeformedDVF = sitk.GetImageFromArray(self._DeformedDVF_, isVector=True)
            DeformedDVF.SetOrigin(self._FixedIm.GetOrigin())
            sitk.WriteImage(sitk.Cast(DeformedDVF, sitk.sitkVectorFloat32), DeforemdDVFAddress)
            DVF_T = sitk.DisplacementFieldTransform(DeformedDVF)    # After this line you cannot save DeformedDVF any more !!!!!!!!!
            DefomedImClean = sitk.Resample(self._FixedIm, DVF_T)          # This is the clean version of the deformed image. Intensity noise should be added to this image
            DefomedIm = sitk.AdditiveGaussianNoise(DefomedImClean, self._setting['sigmaN'], 0, 0)
            self._DefomedIm_=sitk.GetArrayFromImage(DefomedIm)
            sitk.WriteImage(sitk.Cast(DefomedIm, sitk.sitkInt16), DeforemdImAddress)
    #            if (Dsmooth==0):
    #                for Ini in range (1,DsmoothTot):
    #                    start_time = time.time()
    #                    md.smooth(Ginfo,IN,self._FixedIm,FixedMask,FixedImCanny,copy.deepcopy(IncludeMask_),BorderMask_,DeformPath,MaxDeform[D],DistanceDeform,DistanceArea,Np[D],sigmaB[D],sigmaN,sigmaNL,Dsmooth,D,Dim,IncludeMaskErode_,Ini)
    #                    end_time=time.time()
    #                    print( 'IN = '+str(IN)+' Dsmooth = '+str(Dsmooth)+' D = '+str(D)+' Ini = '+str(Ini)+' is Done in {:.3f}s'.format(end_time - start_time))
        end_time = time.time();
        if self._setting['verbose']:
            print( 'SyntheticDeformation: IN = '+str(self._IN)+' Dsmooth = '+str(Dsmooth)+' D = '+str(D)+' is Done in {:.3f}s'.format(end_time - start_time))

        return self._FixedIm_, self._DefomedIm_, self._DeformedDVF_

    def smooth(self):
        Dsmooth = self._Dsmooth
        D = self._D
        DeformTotNames=[self._DeformName+'F',self._DeformName+'M']
        typeTot=['Fixed','Moving']
        DeformFolder=DeformTotNames[self._ImageType]
        DeformPath=self._setting['DLFolder']+'Elastix/'+DeformFolder+'/'
        ExpN = self._Ginfo['ExpN'] + str(self._IN)

        # (DataPath, FolderExp, RootPath, ExpN, FixedImage, MovingImage,  NameF, RootPathDL) = PF.IniFun(self._Ginfo, self._IN)
        MaxDeform = self._setting['MaxDeform'][D]
        Np = self._setting['Np'][D]
        sigmaB = self._setting['sigmaB'][D]
        Border = self._setting['Border']
        Dim = self._setting['Dim']

        if self._Ini > 0:
            Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/DIni' + str(self._Ini) + '/'
        else:
            Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/'
        if not os.path.exists(Dfolder):
            os.makedirs(Dfolder)
        DVFX = np.zeros(self._FixedIm_.shape, dtype=np.float64)
        DVFY = np.zeros(self._FixedIm_.shape, dtype=np.float64)
        DVFZ = np.zeros(self._FixedIm_.shape, dtype=np.float64)

        BorderMask_=np.zeros(self._FixedIm_.shape)
        BorderMask_[Border:self._FixedIm_.shape[0]-Border+1,Border:self._FixedIm_.shape[1]-Border+1,Border:self._FixedIm_.shape[2]-Border+1]=1

        i = 0;
        IEdge = np.where((BorderMask_ > 0) ) # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
        while ((len(IEdge[0]) > 4) & (i < Np)):
            if sys.version_info[0] < 3:
                selectVoxel=long(np.random.randint(0,len(IEdge[0])-1,1,dtype=np.int64))
            else:
                selectVoxel = int(np.random.randint(0, len(IEdge[0]) - 1, 1, dtype=np.int64))

            z = IEdge[0][selectVoxel]
            y = IEdge[1][selectVoxel]
            x = IEdge[2][selectVoxel]
            if i < 2:  # We like to include zero deformation in our training set.
                Dx = 0
                Dy = 0
                Dz = 0
            else:
                Dx = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
                Dy = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
                Dz = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2

            DVFX[z, y, x] = Dx
            DVFY[z, y, x] = Dy
            if (Dim == '3D'):
                DVFZ[z, y, x] = Dz
            else:
                # Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction.
                DVFZ[z, y, x] = 0
            i += 1

        start_time = time.time()
        DVFXb = gaussian_filter(DVFX, sigma=sigmaB)
        end_time = time.time()
        # print(' Smoothing Xdir Done in {:.3f}s'.format(end_time - start_time))
        start_time = time.time()
        DVFYb = gaussian_filter(DVFY, sigma=sigmaB)
        end_time = time.time()
        # print(' Smoothing Ydir Done in {:.3f}s'.format(end_time - start_time))
        if (Dim == '3D'):
            DVFZb = gaussian_filter(DVFZ, sigma=sigmaB)

        IXp = np.where(DVFXb > 0)
        IXn = np.where(DVFXb < 0)
        IYp = np.where(DVFYb > 0)
        IYn = np.where(DVFYb < 0)

        # In the following code, we linearly normalize the DVF for negative and positive values. Please note that if normalization is done for all values,
        # then a shift can occur which leads to many nonzero values
        DVFXb[IXp] = (
        (np.max(DVFX) - 0) / (np.max(DVFXb[IXp]) - np.min(DVFXb[IXp])) * (DVFXb[IXp] - np.min(DVFXb[IXp])) + 0)
        DVFXb[IXn] = (
        (0 - np.min(DVFX[IXn])) / (0 - np.min(DVFXb[IXn])) * (DVFXb[IXn] - np.min(DVFXb[IXn])) + np.min(DVFX[IXn]))
        DVFYb[IYp] = (
        (np.max(DVFY) - 0) / (np.max(DVFYb[IYp]) - np.min(DVFYb[IYp])) * (DVFYb[IYp] - np.min(DVFYb[IYp])) + 0)
        DVFYb[IYn] = (
        (0 - np.min(DVFY[IYn])) / (0 - np.min(DVFYb[IYn])) * (DVFYb[IYn] - np.min(DVFYb[IYn])) + np.min(DVFY[IYn]))

        self._DeformedDVF_[:, :, :, 0] = DVFXb
        self._DeformedDVF_[:, :, :, 1] = DVFYb

        if (Dim == '3D'):
            IZp = np.where(DVFZb > 0)
            IZn = np.where(DVFZb < 0)
            DVFZb[IZp] = (
            (np.max(DVFZ) - 0) / (np.max(DVFZb[IZp]) - np.min(DVFZb[IZp])) * (DVFZb[IZp] - np.min(DVFZb[IZp])) + 0)
            DVFZb[IZn] = (
            (0 - np.min(DVFZ[IZn])) / (0 - np.min(DVFZb[IZn])) * (DVFZb[IZn] - np.min(DVFZb[IZn])) + np.min(DVFZ[IZn]))
            self._DeformedDVF_[:, :, :, 2] = DVFZb
        # if (Ini == 0):
            # sitk.WriteImage(sitk.Cast(DeformedDVF, sitk.sitkVectorFloat32), Dfolder + 'DeformedDVF.mha')


    def blob(self):
        Dsmooth = self._Dsmooth
        D = self._D
        DeformTotNames=[self._DeformName+'F',self._DeformName+'M']
        typeTot=['Fixed','Moving']
        DeformFolder=DeformTotNames[self._ImageType]
        DeformPath=self._setting['DLFolder']+'Elastix/'+DeformFolder+'/'
        ExpN = self._Ginfo['ExpN'] + str(self._IN)

        # (DataPath, FolderExp, RootPath, ExpN, FixedImage, MovingImage, NameF, RootPathDL) = PF.IniFun(self._Ginfo, self._IN)
        MaxDeform = self._setting['MaxDeform'][D]
        Np = self._setting['Np'][D]
        sigmaB = self._setting['sigmaB'][D]
        Border = self._setting['Border']
        Dim = self._setting['Dim']
        DistanceDeform = self._setting['DistanceDeform']
        DistanceArea = self._setting['DistanceArea']
        if self._Ini > 0:
            Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/DIni' + str(self._Ini) + '/'
        else:
            Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/'
        if not os.path.exists(Dfolder):
            os.makedirs(Dfolder)

        DVFX = np.zeros(self._FixedIm_.shape, dtype=np.float64)
        DVFY = np.zeros(self._FixedIm_.shape, dtype=np.float64)
        DVFZ = np.zeros(self._FixedIm_.shape, dtype=np.float64)
        DeformedArea_ = np.zeros(self._FixedIm_.shape)
        BorderMask_ = np.zeros(self._FixedIm_.shape)
        BorderMask_[Border:self._FixedIm_.shape[0] - Border + 1, Border:self._FixedIm_.shape[1] - Border + 1, Border:self._FixedIm_.shape[2] - Border + 1] = 1

        i = 0;
        IEdge = np.where(BorderMask_ > 0) # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
        if (len(IEdge[0]) == 0):
            print('SyntheticDeformation: We are out of points. Plz change the threshold value of Canny method!!!!! ') # Old method. only edges!

        while ((len(IEdge[0]) > 4) & (i < Np)): # IEdge will change at the end of this while loop!
            if sys.version_info[0] < 3:
                selectVoxel = long(np.random.randint(0, len(IEdge[0]) - 1, 1, dtype=np.int64))
            else:
                selectVoxel = int(np.random.randint(0, len(IEdge[0]) - 1, 1, dtype=np.int64))
            z = IEdge[0][selectVoxel]
            y = IEdge[1][selectVoxel]
            x = IEdge[2][selectVoxel]
            if i < 2:  # We like to include zero deformation in our training set.
                Dx = 0
                Dy = 0
                Dz = 0
            else:
                Dx = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
                Dy = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
                Dz = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2

            DVFX[z, y, x] = Dx
            DVFY[z, y, x] = Dy
            DVFZ[z, y, x] = Dz

            xminD = x - DistanceDeform
            xmaxD = x + DistanceDeform
            yminD = y - DistanceDeform
            ymaxD = y + DistanceDeform
            zminD = z - DistanceDeform
            zmaxD = z + DistanceDeform

            if zmaxD > (self._FixedIm_.shape[0] - 1): zmaxD = (self._FixedIm_.shape[0] - 1)
            if ymaxD > (self._FixedIm_.shape[1] - 1): ymaxD = (self._FixedIm_.shape[1] - 1)
            if xmaxD > (self._FixedIm_.shape[2] - 1): xmaxD = (self._FixedIm_.shape[2] - 1)
            if zminD < 0: zminD = 0
            if yminD < 0: yminD = 0
            if xminD < 0: xminD = 0
            xminA = x - DistanceArea
            xmaxA = x + DistanceArea
            yminA = y - DistanceArea
            ymaxA = y + DistanceArea
            if (Dim == '3D'):
                zminA = z - DistanceArea
                zmaxA = z + DistanceArea
            else:
                zminA = z - 1
                zmaxA = z + 2  # This is exclusively for 2D !!!!

            if zmaxA > (self._FixedIm_.shape[0] - 1): zmaxA = (self._FixedIm_.shape[0] - 1)
            if ymaxA > (self._FixedIm_.shape[1] - 1): ymaxA = (self._FixedIm_.shape[1] - 1)
            if xmaxA > (self._FixedIm_.shape[2] - 1): xmaxA = (self._FixedIm_.shape[2] - 1)
            if zminA < 0: zminA = 0
            if yminA < 0: yminA = 0
            if xminA < 0: xminA = 0

            BorderMask_[zminD:zmaxD, yminD:ymaxD, xminD:xmaxD] = 0
            DeformedArea_[zminA:zmaxA, yminA:ymaxA, xminA:xmaxA] = 1
            IEdge = np.where(BorderMask_ > 0)
            i += 1
        del BorderMask_

        DeformedArea = sitk.GetImageFromArray(DeformedArea_)
        DeformedArea.SetOrigin(self._FixedIm.GetOrigin())
        DeformedArea.SetSpacing(self._FixedIm.GetSpacing())
        sitk.WriteImage(DeformedArea, Dfolder + 'DeformedArea.mha')

        DVFXb = gaussian_filter(DVFX, sigma=sigmaB)
        DVFYb = gaussian_filter(DVFY, sigma=sigmaB)
        DVFZb = gaussian_filter(DVFZ, sigma=sigmaB)

        IXp = np.where(DVFXb > 0)
        IXn = np.where(DVFXb < 0)
        IYp = np.where(DVFYb > 0)
        IYn = np.where(DVFYb < 0)
        IZp = np.where(DVFZb > 0)
        IZn = np.where(DVFZb < 0)

        DVFXb[IXp] = ((np.max(DVFX) - 0) / (np.max(DVFXb[IXp]) - np.min(DVFXb[IXp])) * (DVFXb[IXp] - np.min(DVFXb[IXp])) + 0)
        DVFXb[IXn] = ((0 - np.min(DVFX[IXn])) / (0 - np.min(DVFXb[IXn])) * (DVFXb[IXn] - np.min(DVFXb[IXn])) + np.min(DVFX[IXn]))
        DVFYb[IYp] = ((np.max(DVFY) - 0) / (np.max(DVFYb[IYp]) - np.min(DVFYb[IYp])) * (DVFYb[IYp] - np.min(DVFYb[IYp])) + 0)
        DVFYb[IYn] = ((0 - np.min(DVFY[IYn])) / (0 - np.min(DVFYb[IYn])) * (DVFYb[IYn] - np.min(DVFYb[IYn])) + np.min(DVFY[IYn]))

        self._DeformedDVF_[:, :, :, 0] = DVFXb
        self._DeformedDVF_[:, :, :, 1] = DVFYb

        if (Dim == '3D'):
            DVFZb[IZp] = ((np.max(DVFZ) - 0) / (np.max(DVFZb[IZp]) - np.min(DVFZb[IZp])) * (DVFZb[IZp] - np.min(DVFZb[IZp])) + 0)
            DVFZb[IZn] = ((0 - np.min(DVFZ[IZn])) / (0 - np.min(DVFZb[IZn])) * (DVFZb[IZn] - np.min(DVFZb[IZn])) + np.min(DVFZ[IZn]))
            self._DeformedDVF_[:, :, :, 2] = DVFZb


# def saveCanny(FixedImCanny,Ginfo,IN,FixedIm,Dsmooth,DeformPath):
#     (DataPath,FolderExp,RootPath,ExpN,FixedImage,MovingImage,NameF,RootPathDL)= PF.IniFun(self._Ginfo,IN)
#     if Ini>0:
#         Dfolder=DeformPath+ExpN+'/Dsmooth'+str(Dsmooth)+'/DIni'+str(Ini)+'/'
#     else:
#         Dfolder=DeformPath+ExpN+'/Dsmooth'+str(Dsmooth)+'/D'+str(D)+'/'
