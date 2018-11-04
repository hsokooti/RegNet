import numpy as np
import SimpleITK as sitk
import os, time, sys
import _pickle as pickle
import matplotlib.pyplot as plt
from importlib import reload
import multiprocessing
import Functions.SyntheticDeformation as syndef

# %%-------------------------------------------h.sokooti@gmail.com--------------------------------------------

def searchIndices(DVFList1, c, classBalanced, K, Dim):
    '''
    This function searches for voxels based on the classBalanced in the parallel mode: if Setting['ParallelSearch'] == True

    :param DVFList1: input DVF
    :param c:               enumerate of the class (in for loop over all classes)
    :param classBalanced:   a vector indicates the classes, for instance [a,b] implies classes [0,a), [a,b)
    :param K:               Margin of the image. so no voxel would be selected if the index is smaller than K or greater than (ImageSize - K)
    :param Dim:             '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction. Hence, we can't use np.all and we have to use np.any.

    :return:                I1 which is a numpy array of ravel_multi_index

    Hessam Sokooti h.sokooti@gmail.com
    '''
    gridDVF = np.transpose(np.indices(np.shape(DVFList1)[:-1]), (1, 2, 3, 0))
    if c == 0:
        # Future: you can add a mask here to prevent selecting pixels twice!
        I1 = np.ravel_multi_index(np.where((np.all((np.abs(DVFList1) < classBalanced[c]), axis=3)) &
                                           (np.all((gridDVF > K), axis=3)) & (np.all((gridDVF < [np.array(np.shape(DVFList1)[:-1]) - K]), axis=3))), np.shape(DVFList1)[:-1]).astype(np.int32)
        # the output of np.where occupy huge part of memory! by converting it to a numpy array lots of memory can be saved!
    if (c > 0) & (c < len(classBalanced)):
        if Dim == '2D':
            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
            I1 = np.ravel_multi_index(np.where((np.all((np.abs(DVFList1) < classBalanced[c]), axis=3)) & (np.any((np.abs(DVFList1) >= classBalanced[c - 1]), axis=3)) &
                                           (np.all((gridDVF > K), axis=3)) & (np.all((gridDVF < [np.array(np.shape(DVFList1)[:-1]) - K]), axis=3))), np.shape(DVFList1)[:-1]).astype(np.int32)
        if Dim == '3D':
            I1 = np.ravel_multi_index(np.where((np.all((np.abs(DVFList1) < classBalanced[c]), axis=3)) & (np.all((np.abs(DVFList1) >= classBalanced[c - 1]), axis=3)) &
                                               (np.all((gridDVF > K), axis=3)) & (np.all((gridDVF < [np.array(np.shape(DVFList1)[:-1]) - K]), axis=3))), np.shape(DVFList1)[:-1]).astype(np.int32)
    # if c == len(classBalanced):
    #     I1 = np.ravel_multi_index(np.where((np.any((np.abs(DVFList1) >= classBalanced[c - 1]), axis=3)) &
    #                                        (np.all((gridDVF > K), axis=3)) & (np.all((gridDVF < [np.array(np.shape(DVFList1)[:-1]) - K]), axis=3))), np.shape(DVFList1)[:-1]).astype(np.int32)
    return I1

class Patches(object):
    def __init__(self,               
               setting ={},
               numberOfImagesPerChunk= 5,    # number of images that I would like to load in RAM
               samplesPerImage = 1500,
               IN = np.arange(1, 11) ,
               training = 0
               ):

        self._setting = setting
        self._numberOfImagesPerChunk= numberOfImagesPerChunk
        self._samplesPerImage = samplesPerImage
        self._batchCounter=0
      
        self._chunk = 0
        self._chunks_completed = 0
        self._semiEpochs_completed = 0
        self._semiEpoch = 0
        self._batchCounter = 0
        self._FixedImList = [None]*numberOfImagesPerChunk
        self._DeformedImList=[None]*numberOfImagesPerChunk
        self._DVFList= [None]*numberOfImagesPerChunk
        self._IN = IN
        self._training = training

    def fillList(self):
        numberOfImagesPerChunk = self._numberOfImagesPerChunk
        if self._training :
            # Make all lists empty in training mode. In the test mode, we keep the same chunk forever. So no need to make it empty and refill it again.
            self._FixedImList = [None]*numberOfImagesPerChunk
            self._DeformedImList=[None]*numberOfImagesPerChunk
            self._DVFList= [None]*numberOfImagesPerChunk
        
        if self._semiEpochs_completed:
            self._semiEpoch = self._semiEpoch + 1
            self._semiEpochs_completed = 0
            self._chunk = 0
 
        Dsmooth=0 
        deformMethod = self._setting['deformMethod']
        IN = self._IN      # IN: Index Number of the images
        ImageTypeTotal = np.repeat([0,1], len(IN)*len(deformMethod))  # 0 for fixed images and 1 for moving images
        INTotal = np.tile(np.repeat(IN,len(deformMethod)), 2) # Making image numbers for all fixed, moving and all deform methods (low freq, medium freq, high freq.)
        DTotal  = np.tile(deformMethod, len(IN) *2)                
        np.random.seed(self._semiEpoch)
        randomPairTotal = np.random.permutation(len(INTotal))

        lowerRange = ((self._chunk) * numberOfImagesPerChunk)
        upperRange = ((self._chunk+1)*self._numberOfImagesPerChunk)
        if upperRange >= len(INTotal):
            upperRange = len(INTotal)
            self._semiEpochs_completed = 1            
            numberOfImagesPerChunk = upperRange - lowerRange # In cases when last chunk of images are smaller than the self._numberOfImagesPerChunk
            self._FixedImList = [None]*numberOfImagesPerChunk
            self._DeformedImList=[None]*numberOfImagesPerChunk
            self._DVFList= [None]*numberOfImagesPerChunk

        pairSelection= randomPairTotal[lowerRange : upperRange]
        for i, pair in enumerate(pairSelection):
            SyntheticDVF = syndef.SyntheticDVF( setting = self._setting,
                ImageType = ImageTypeTotal[pair],  # 0: Fixed image, 1: Moving image
                IN = INTotal[pair],             # number of the image in the database. In SPREAD database it can be between 1 and 21. (Please note that it starts from 1 not 0)
                DeformName = self._setting['deformName'],  # Name of the folder to write or to read.
                Dsmooth = Dsmooth,              # This variable is used to generate another deformed version of the moving image. Then, use that image to make synthetic DVFs. More information available on [sokooti2017nonrigid]
                D = DTotal[pair]                # 0: low freq, 1: medium freq, 2: high freq. More information available on [sokooti2017nonrigid])
                                     )
            self._FixedImList[i], self._DeformedImList[i], self._DVFList[i] = SyntheticDVF.GetDVFandDeformedImages()
            if self._setting['verbose']:
                print ('RegNet: Image type = {1:} IN = {1:} is loaded'.format(ImageTypeTotal[pair], INTotal[pair] ))
        K=self._setting['K']
        classBalanced = self._setting['classBalanced']
        indices={} ; 
        start_time = time.time() 

        if self._setting['ParallelSearch']:
            from joblib import Parallel, delayed
            num_cores = multiprocessing.cpu_count() - 2
            start_time = time.time()
            results = Parallel(n_jobs=num_cores)(delayed(searchIndices)(DVFList1=self._DVFList[i], c = c, classBalanced= classBalanced, K = K , Dim = self._setting['Dim'])
                                             for i in range(0, len(self._FixedImList)) for c in range(0, len(classBalanced)))

            indices = {}
            for iresults in range(0, len(results)):
                i = iresults // (len(classBalanced) )  # first loop in the Parallel: for i in range(0, len(FixedImList))
                c = iresults % (len(classBalanced) )  # second loop in the Parallel: for j in range(0, len(classBalanced)+1)
                # print(' i = {} c = {}'.format(i, c))
                if (i == 0) or (len(indices['class' + str(c)]) == 0):
                    indices['class' + str(c)] = np.array(np.c_[results[iresults], i * np.ones(len(results[iresults]), dtype=np.int32)])
                else:
                    indices['class' + str(c)] = np.concatenate((indices['class' + str(c)], np.array(np.c_[results[iresults], i * np.ones(len(results[iresults]), dtype=np.int32)])), axis=0)
            del results
            end_time = time.time();
            if self._setting['verbose']:
                print('RegNet: Parallel searching for {} classes is Done in {:.2f}s'.format(len(classBalanced), end_time - start_time))
        else:
            for i in range (0,len(self._FixedImList)) :
                gridDVF = np.transpose(np.indices(np.shape(self._DVFList[i])[:-1]),(1,2,3,0))
                for c in range (0, len(self._classBalanced) ):
                    if c == 0:
                        # you can add a mask here to prevent selecting pixels twice!
                        I1 = np.ravel_multi_index(np.where( (np.all((np.abs(self._DVFList[i]) < classBalanced[c]),axis=3)) &
                               (np.all((gridDVF > K),axis = 3)) & (np.all((gridDVF < [ np.array(np.shape(self._DVFList[i])[:-1]) - K]), axis = 3 ))) ,
                                np.shape(self._DVFList[i])[:-1] ).astype(np.int32)
                        # the output of np.where occupy huge part of memory! by converting it to a numpy array lots of memory can be saved!
                    if (c > 0 ) & (c < (len(classBalanced))):
                        if self._setting['Dim'] == '2D':
                            # in 2D experiments, the DVFList is still in 3D and for the third direction is set to 0. Here we use np.any() instead of np.all()
                            I1 = np.ravel_multi_index(np.where( (np.all((np.abs(self._DVFList[i]) < classBalanced[c]),axis=3)) &
                                   (np.any((np.abs(self._DVFList[i]) >= classBalanced[c-1]),axis=3)) & (np.all((gridDVF > K),axis = 3)) &
                                   (np.all((gridDVF < [ np.array(np.shape(self._DVFList[i])[:-1]) - K]), axis = 3 ))) , np.shape(self._DVFList[i])[:-1] ).astype(np.int32)
                        if self._setting['Dim'] == '3D':
                            I1 = np.ravel_multi_index(np.where( (np.all((np.abs(self._DVFList[i]) < classBalanced[c]),axis=3)) &
                                   (np.all((np.abs(self._DVFList[i]) >= classBalanced[c-1]),axis=3)) & (np.all((gridDVF > K),axis = 3)) &
                                   (np.all((gridDVF < [ np.array(np.shape(self._DVFList[i])[:-1]) - K]), axis = 3 ))) , np.shape(self._DVFList[i])[:-1] ).astype(np.int32)
                    # if c == len(classBalanced) :
                    #     I1 = np.ravel_multi_index(np.where( (np.any((np.abs(self._DVFList[i]) >= classBalanced[c]),axis=3)) &
                    #            (np.all((gridDVF > K),axis = 3)) & (np.all((gridDVF < [ np.array(np.shape(self._DVFList[i])[:-1]) - K]), axis = 3 )))
                    #             , np.shape(self._DVFList[i])[:-1] ).astype(np.int32)
                    if (i == 0)  or (len (indices['class'+str(c)] )== 0) :
                        indices['class'+str(c)] =np.array( np.c_[I1 , i * np.ones(len(I1),dtype=np.int32)])
                    else:
                        indices['class'+str(c)] = np.concatenate(( indices['class'+str(c)] , np.array( np.c_[I1 , i * np.ones(len(I1), dtype = np.int32)])), axis = 0)
                    if self._setting['verbose']:
                        print( 'Finding classes done for i = {}, c = {} '.format(i, c) )
            del I1; end_time = time.time();
            if self._setting['verbose']:
                print( 'Searching for {} classes is Done in {:.2f}s'.format(len(classBalanced) +1, end_time - start_time))
        samplesPerChunk = self._samplesPerImage *  numberOfImagesPerChunk
        SamplePerChunkPerClass = np.round(samplesPerChunk / (len (classBalanced)))
        numberSamplesClass = np.empty(len(classBalanced),dtype = np.int32)
        np.random.seed(self._semiEpoch*1000+self._chunk)
        for c,k  in enumerate(indices.keys()):    
            numberSamplesClass[c] = min (SamplePerChunkPerClass, np.shape(indices[k])[0])  
            # it is possible to have different number in each class. However we perefer to have at least SamplePerChunkPerClass            
            I1 = np.random.randint(0 , high= np.shape(indices['class'+str(c)])[0], size = numberSamplesClass[c])
            if c == 0:
                I =np.concatenate(( indices['class'+str(c)][I1,:] , c * np.ones([len(I1),1], dtype = np.int32)),axis = 1). astype (np.int32)
            else:
                I = np.concatenate ((I ,  np.concatenate(( indices['class'+str(c)][I1,:] , c * np.ones([len(I1),1], dtype = np.int32)),axis = 1) ), axis =0 )
        if self._setting['verbose']:
            print( 'RegNet: samplesPerChunk is {} for semiEpoch = {}, Chunk = {} '.format(sum(numberSamplesClass), self._semiEpoch , self._chunk) )
        shuffleIndex= np.arange(0, len(I))
        np.random.shuffle(shuffleIndex)
        self._Ish = I[shuffleIndex]  # Ish : Shuffled Index

    def goToNextChunk(self):
        if self._semiEpochs_completed:
            self._semiEpoch = self._semiEpoch + 1
            self._semiEpochs_completed = 0
            self._chunk = 0  #
        else:
            self._chunk = self._chunk + 1
            # upperRange = ((self._chunk+1)*self._numberOfImagesPerChunk)
            # if upperRange >= len(INTotal):
            #     self._semiEpochs_completed = 1
            # self._batchCounter = 0
            # self._chunks_completed = 0
        print('RegNet: NextChunk, is_training = {} semiEpoch = {}, Chunk = {}, batchCounter = {} , endBatch = {} '.format(self._training, self._semiEpoch, self._chunk, self._batchCounter, endBatch))


    def next_batch(self,batchSize):
        
        if self._chunks_completed:
            self._chunk = self._chunk + 1
            self.fillList()
            self._batchCounter = 0 
            self._chunks_completed = 0
                            
        R = self._setting['R']
        Ry = self._setting['Ry']
        Rlow = self._setting['Rlow']
        Ish = self._Ish
        if R > self._setting['Border']:
            hi =1
            # raise ValueError ( 'R = {} should be smaller than Setting[Borders] ={}  '.format(R, self._setting['Border']))
        endBatch = (self._batchCounter+1)*batchSize
        if endBatch >= len (Ish):
            self._chunks_completed = 1
            endBatch = len (Ish)

        # Ish [: , 0] the index of the sample that is gotten from np.where
        # Ish [: , 1] the the number of the image in self._FixedImList
        # Ish [: , 2] the the number of class, which is not needed anymore!!

        if self._setting['Dim'] == '2D':
            batchXlow = 0
            BatchXFixed=np.stack([self._FixedImList[Ish[i,1]][
                    np.unravel_index(Ish[i,0] , np.shape(self._FixedImList[Ish[i,1]]))[0] ,  
                    np.unravel_index(Ish[i,0] , np.shape(self._FixedImList[Ish[i,1]]))[1] - R :np.unravel_index(Ish[i,0] , np.shape(self._FixedImList[Ish[i,1]]))[1] + R + 1, 
                    np.unravel_index(Ish[i,0] , np.shape(self._FixedImList[Ish[i,1]]))[2] - R :np.unravel_index(Ish[i,0] , np.shape(self._FixedImList[Ish[i,1]]))[2] + R  + 1,
                    np.newaxis] for i in range(self._batchCounter*batchSize, endBatch)])
            BatchXDeformed=np.stack([self._DeformedImList[Ish[i,1]][
                    np.unravel_index(Ish[i,0] , np.shape(self._DeformedImList[Ish[i,1]]))[0] ,  
                    np.unravel_index(Ish[i,0] , np.shape(self._DeformedImList[Ish[i,1]]))[1] - R :np.unravel_index(Ish[i,0] , np.shape(self._DeformedImList[Ish[i,1]]))[1] + R + 1, 
                    np.unravel_index(Ish[i,0] , np.shape(self._DeformedImList[Ish[i,1]]))[2] - R :np.unravel_index(Ish[i,0] , np.shape(self._DeformedImList[Ish[i,1]]))[2] + R  + 1,
                    np.newaxis] for i in range(self._batchCounter*batchSize, endBatch)])    

            if self._setting['Resolution'] == 'multi':
                BatchXFixed_LowRes = np.stack([self._FixedImList[Ish[i, 1]][
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0],
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] + Rlow + 1:2,
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] + Rlow + 1:2,
                                    np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])
                BatchXDeformed_LowRes = np.stack([self._DeformedImList[Ish[i, 1]][
                                   np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[0],
                                   np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] + Rlow + 1:2,
                                   np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] + Rlow + 1:2,
                                   np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])
                batchXlow = np.concatenate((BatchXFixed_LowRes, BatchXDeformed_LowRes), axis=3).astype(np.float32)

            batchX = np.concatenate((BatchXFixed, BatchXDeformed), axis=3).astype(np.float32)
            batchY = np.stack([self._DVFList[Ish[i, 1]][
                           np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0],
                           np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] - Ry:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] + Ry + 1,
                           np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] - Ry:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] + Ry + 1,
                           0:2] for i in range(self._batchCounter * batchSize, endBatch)])
        if self._setting['Dim'] == '3D':
            batchXlow = 0
            BatchXFixed = np.stack([self._FixedImList[Ish[i, 1]][
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] - R:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] + R + 1,
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] - R:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] + R + 1,
                                    np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] - R:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] + R + 1,
                                    np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])
            BatchXDeformed = np.stack([self._DeformedImList[Ish[i, 1]][
                                       np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[0] - R:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[0] + R + 1,
                                       np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] - R:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] + R + 1,
                                       np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] - R:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] + R + 1,
                                       np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])

            if self._setting['Resolution'] == 'multi':
                BatchXFixed_LowRes = np.stack([self._FixedImList[Ish[i, 1]][
                            np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] + Rlow + 1 : 2,
                            np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] + Rlow + 1 : 2,
                            np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] + Rlow + 1 : 2,
                            np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])
                BatchXDeformed_LowRes = np.stack([self._DeformedImList[Ish[i, 1]][
                           np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[0] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[0] + Rlow + 1 : 2,
                           np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[1] + Rlow + 1 : 2,
                           np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] - Rlow:np.unravel_index(Ish[i, 0], np.shape(self._DeformedImList[Ish[i, 1]]))[2] + Rlow + 1 : 2,
                           np.newaxis] for i in range(self._batchCounter * batchSize, endBatch)])
                batchXlow = np.concatenate((BatchXFixed_LowRes, BatchXDeformed_LowRes), axis=4).astype(np.float32)

            batchX = np.concatenate((BatchXFixed, BatchXDeformed), axis=4).astype(np.float32)
            batchY = np.stack([self._DVFList[Ish[i, 1]][
                               np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] - Ry:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[0] + Ry + 1,
                               np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] - Ry:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[1] + Ry + 1,
                               np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] - Ry:np.unravel_index(Ish[i, 0], np.shape(self._FixedImList[Ish[i, 1]]))[2] + Ry + 1,
                               0:3] for i in range(self._batchCounter * batchSize, endBatch)])
        if self._setting['verbose']:
            print( 'is_training = {} semiEpoch = {}, Chunk = {}, batchCounter = {} , endBatch = {} '.format(self._training, self._semiEpoch , self._chunk, self._batchCounter, endBatch) )
        self._batchCounter = self._batchCounter + 1
        return batchX, batchY, batchXlow

    def resetValidation(self):
        self._batchCounter = 0 
        self._chunks_completed = 0
        self._chunk= 0
        self._semiEpochs_completed = 0
        self._semiEpoch = 0

    def copyFromThread(self, PatchesThread):
        self._setting = PatchesThread._setting
        self._numberOfImagesPerChunk = PatchesThread._numberOfImagesPerChunk
        self._samplesPerImage = PatchesThread._samplesPerImage
        self._batchCounter = 0
        self._chunk = PatchesThread._chunk
        self._chunks_completed = PatchesThread._chunks_completed
        self._semiEpochs_completed = PatchesThread._semiEpochs_completed
        self._semiEpoch = PatchesThread._semiEpoch
        self._FixedImList = PatchesThread._FixedImList
        self._DeformedImList = PatchesThread._DeformedImList
        self._DVFList = PatchesThread._DVFList
        self._IN = PatchesThread._IN
        self._traininRegNetTraing = PatchesThread._training
        self._Ish = PatchesThread._Ish