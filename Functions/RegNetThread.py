import numpy as np
import SimpleITK as sitk
import os, time, sys
import _pickle as pickle
import matplotlib.pyplot as plt
from importlib import reload
import multiprocessing , threading
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
        # you can add a mask here to prevent selecting pixels twice!
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



class PatchesThread(threading.Thread):
    def __init__(self,
               setting ={},
               numberOfImagesPerChunk= 5,    # number of images that I would like to load in RAM
               samplesPerImage = 1500,
               IN = np.arange(1, 11) ,
               training = 0
               ):
        threading.Thread.__init__(self)
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.daemon = True

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

        self._filled = 0


    def run(self):
        # Borrowed from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                self.fillList()
            time.sleep(5)

    def pause(self):
        # Modified from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        if not self.paused:
            self.paused = True
            # If in sleep, we acquire immediately, otherwise we wait for thread
            # to release condition. In race, worker will still see self.paused
            # and begin waiting until it's set back to False
            self.pause_cond.acquire()

    def resume(self):
        # Modified from: https://stackoverflow.com/questions/33640283/python-thread-that-i-can-pause-and-resume
        if self.paused:
            self.paused = False
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()

    def fillList(self):
        self._filled = 0
        numberOfImagesPerChunk  = self._numberOfImagesPerChunk
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
        self._INTotal = INTotal
        DTotal  = np.tile(deformMethod, len(IN) *2)
        np.random.seed(self._semiEpoch)
        randomPairTotal = np.random.permutation(len(INTotal))                 

        lowerRange = ((self._chunk)*numberOfImagesPerChunk)
        upperRange = ((self._chunk+1)*numberOfImagesPerChunk)
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
                print ('Thread: Image type = {1:} IN = {1:} is loaded'.format(ImageTypeTotal[pair], INTotal[pair] ))
        K=self._setting['K']
        classBalanced = self._setting['classBalanced']
        indices={} ; 
        start_time = time.time() 

        if self._setting['ParallelSearch']:
            from joblib import Parallel, delayed
            num_cores = multiprocessing.cpu_count() - 2
            start_time = time.time()
            results = Parallel(n_jobs=num_cores)(delayed(searchIndices)(DVFList1=self._DVFList[i], c = c, classBalanced= classBalanced, K = K,  Dim = self._setting['Dim'])
                                                 for i in range(0, len(self._FixedImList)) for c in range(0, len(classBalanced)))

            indices = {}
            for iresults in range(0, len(results)):
                i = iresults // (len(classBalanced))  # first loop in the Parallel: for i in range(0, len(FixedImList))
                c = iresults % (len(classBalanced))  # second loop in the Parallel: for j in range(0, len(classBalanced)+1)
                # print(' i = {} c = {}'.format(i, c))
                if (i == 0) or (len(indices['class' + str(c)]) == 0):
                    indices['class' + str(c)] = np.array(np.c_[results[iresults], i * np.ones(len(results[iresults]), dtype=np.int32)])
                else:
                    indices['class' + str(c)] = np.concatenate((indices['class' + str(c)], np.array(np.c_[results[iresults], i * np.ones(len(results[iresults]), dtype=np.int32)])), axis=0)
            del results
            end_time = time.time()
            if self._setting['verbose']:
                print('Thread: Parallel searching for {} classes is Done in {:.2f}s'.format(len(classBalanced), end_time - start_time))
        else:
            for i in range (0,len(self._FixedImList)) :
                gridDVF = np.transpose(np.indices(np.shape(self._DVFList[i])[:-1]),(1,2,3,0))
                for c in range (0, len(self._classBalanced)):
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
                        print( 'Thread: Finding classes done for i = {}, c = {} '.format(i, c) )
            del I1; end_time = time.time(); print( 'Thread Searching for {} classes is Done in {:.2f}s'.format(len(classBalanced) +1, end_time - start_time))
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
            print( 'Thread: samplesPerChunk is {} for semiEpoch = {}, Chunk = {} '.format(sum(numberSamplesClass), self._semiEpoch , self._chunk) )
        shuffleIndex= np.arange(0, len(I))
        np.random.shuffle(shuffleIndex)
        self._Ish = I[shuffleIndex]  # Ish : Shuffled Index
        self._filled = 1
        print ('Thread is filled .....................')
        self.pause()

    def goToNextChunk(self):
        if self._semiEpochs_completed:
            self._semiEpoch = self._semiEpoch + 1
            self._semiEpochs_completed = 0
            self._chunk = 0  #
        else:
            self._chunk = self._chunk + 1
            # upperRange = ((self._chunk+1)*self._numberOfImagesPerChunk)
            # if upperRange >= len(self._INTotal):
            #     self._semiEpochs_completed = 1
            # self._batchCounter = 0
            # self._chunks_completed = 0
        print('Thread: NextChunk, is_training = {} semiEpoch = {}, Chunk = {}, batchCounter = {}  '.format(self._training, self._semiEpoch, self._chunk, self._batchCounter))






