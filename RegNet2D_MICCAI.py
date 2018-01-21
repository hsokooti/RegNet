import numpy as np
import os, time , datetime, sys
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import Functions.RegNet as RegNet
import Functions.RegNetThread as RegNetThread
import Functions.PyFunctions as PF

# %%-------------------------------------------h.sokooti@gmail.com--------------------------------------------

myDate = current_time = datetime.datetime.now()
LOGDIR = '/home/hsokooti/DL/RegNet2/TB/2DB/'
Exp='MICCAI_{:04d}{:02d}{:02d}_{:02d}{:02d}'.format(myDate.year, myDate.month, myDate.day, myDate.hour, myDate.minute, myDate.second)

# saving the current script
if not(os.path.isdir(LOGDIR + 'train' + Exp+'/Model/')):
    os.makedirs(LOGDIR + 'train' + Exp+'/Model/')
    print('folder created')
shutil.copy(os.path.realpath(__file__),LOGDIR + 'train' + Exp+'/Model/')
sys.stdout = PF.Logger(LOGDIR + 'train' + Exp+'/Model/log.txt')

def RegNet_model(learning_rate = 1E-4 , max_steps = 1000):
    tf.reset_default_graph()
    sess = tf.Session()
    with tf.name_scope('inputs') as scope:
        x = tf.placeholder(tf.float32, shape=[None, 29, 29 , 2], name="x")
        # x = tf.placeholder(tf.float32, shape=[None, None, None, 2], name="x")
        xLow = tf.placeholder(tf.float32, shape=[None, 27, 27, 2], name="xLow")
        # tf.summary.image('input', x_image, 3)
        y = tf.placeholder(tf.float32, shape=[None, 1, 1, 2], name="labels")
        # y = tf.placeholder(tf.float32, shape=[None, None, None, 2], name="labels")
        bn_training = tf.placeholder(tf.bool, name='bn_training')
        mseTrainAverage_net = tf.placeholder(tf.float32 , shape = []  )
        y_dirX=y[0,np.newaxis,:,:,0,np.newaxis]
        y_dirY=y[0,np.newaxis,:,:,1,np.newaxis]
        x_Fixed=x[0,np.newaxis,:,:,0,np.newaxis]
        x_Deformed = x[0, np.newaxis, :, :, 1, np.newaxis]
        tf.summary.image('y_dir', tf.concat((y_dirX,y_dirY), 0),2)
        tf.summary.image('Images', tf.concat((x_Fixed,x_Deformed),0), 2)
        # tf.summary.image('x_Deformed', x_Deformed, 1)

    with tf.name_scope('lateFusion') as scope:
        conv1F = tf.layers.conv2d(inputs=x[:,:,:,0,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1F')
        conv1F = tf.layers.batch_normalization(conv1F, training=bn_training,  name='bn1F', scale=True)
        conv1F = tf.nn.relu(conv1F)

        conv1M = tf.layers.conv2d(inputs=x[:,:,:,1,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1M')
        conv1M = tf.layers.batch_normalization(conv1M, training=bn_training,  name='bn1M', scale=True)
        conv1M = tf.nn.relu(conv1M)

        conv1FLow = tf.layers.conv2d(inputs=xLow[:,:,:,0,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1FLow')
        conv1FLow = tf.layers.batch_normalization(conv1FLow, training=bn_training,  name='bn1FLow', scale=True)
        conv1FLow = tf.nn.relu(conv1FLow)

        conv1MLow = tf.layers.conv2d(inputs=xLow[:,:,:,1,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1MLow')
        conv1MLow = tf.layers.batch_normalization(conv1MLow, training=bn_training,  name='bn1MLow', scale=True)
        conv1MLow = tf.nn.relu(conv1MLow)

        for i in range (2,4):
            conv1F = tf.layers.conv2d(conv1F, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'F')
            conv1F = tf.layers.batch_normalization(conv1F, training=bn_training)
            conv1F = tf.nn.relu(conv1F)

        for i in range(2, 4):
            conv1M = tf.layers.conv2d(conv1M, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'M')
            conv1M = tf.layers.batch_normalization(conv1M, training=bn_training)
            conv1M = tf.nn.relu(conv1M)

        for i in range (2,4):
            conv1FLow = tf.layers.conv2d(conv1FLow, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'FLow')
            conv1FLow = tf.layers.batch_normalization(conv1FLow, training=bn_training)
            conv1FLow = tf.nn.relu(conv1FLow)

        for i in range (2,4):
            conv1MLow = tf.layers.conv2d(conv1MLow, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'MLow')
            conv1MLow = tf.layers.batch_normalization(conv1MLow, training=bn_training)
            conv1MLow = tf.nn.relu(conv1MLow)

    with tf.name_scope('MergeFixedMoving') as scope:
        conv2 = tf.concat([conv1F, conv1M], 3)
        conv2Low = tf.concat([conv1FLow, conv1MLow], 3)

    numberOfFeatures = [25,25,25,30,30,30]
    for i in range (4,10):
        conv2Low = tf.layers.conv2d(conv2Low, numberOfFeatures[i-4] , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'Low')
        conv2Low = tf.layers.batch_normalization(conv2Low, training=bn_training)
        conv2Low = tf.nn.relu(conv2Low)

    numberOfFeatures = [25,30]
    for i in range(4, 6):
        conv2 = tf.layers.conv2d(conv2, numberOfFeatures[i-4] , [3, 3],  padding="valid", activation=None, name='conv'+str(i))
        conv2 = tf.layers.batch_normalization(conv2, training=bn_training)
        conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.max_pooling2d(conv2 , [2,2], 2, name='conv6')

    conv3 = tf.concat([conv2, conv2Low], 3)

    numberOfFeatures = [60, 70, 75, 150]
    for i in range(1, 5):
        conv3 = tf.layers.conv2d(conv3, numberOfFeatures[i-1] , [3, 3],  padding="valid", activation=None, name='convFullyConnected'+str(i))
        conv3 = tf.layers.batch_normalization(conv3, training=bn_training)
        conv3 = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d(conv3, numberOfFeatures[i-1] , [1, 1],  padding="valid", activation=None, name='convFullyConnected'+str(5))
    conv4 = tf.layers.batch_normalization(conv4, training=bn_training)
    conv4 = tf.nn.relu(conv4)

    yHat = tf.layers.conv2d(conv4, 2, [1, 1], padding="valid", activation=None, dilation_rate=(1, 1))
    mse = (tf.losses.huber_loss(y, yHat, weights=1))

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    print('mse shape %s ' % (mse.get_shape()))
    print('y shape %s ' % (y.get_shape()))
    yHat_dirX=yHat[0,np.newaxis,:,:,0,np.newaxis]
    yHat_dirY=yHat[0,np.newaxis,:,:,1,np.newaxis]
    tf.summary.image('yHat_dir', tf.concat((yHat_dirX, yHat_dirY), 0), 2)
    tf.summary.scalar("mse", mseTrainAverage_net)

    summ = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOGDIR + '/train'+Exp, sess.graph)
    test_writer = tf.summary.FileWriter(LOGDIR + '/test' + Exp, sess.graph)
    # tf.global_variables_initializer().run() #Otherwise you encounter this error : Attempting to use uninitialized value conv2d/kerne
    print(' total numbe of variables %s' %(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # %%------------------------------------------- Setting of generating synthetic DVFs------------------------------------------
    Setting = {}
    Setting['DLFolder'] = '/srv/2-lkeb-16-reg1/hsokooti/DL/'    # 'E:/PHD/Software/Project/DL/' or '/srv/2-lkeb-16-reg1/hsokooti/DL/'
    Setting['deformName'] = 'LungExp2D_1'            # 2D: 'LungExp2D_1'   , 3D: 'LungExp3D_2'
    Setting['Dim'] = '2D'               # '2D' or '3D'. Please note that in 2D setting, we still have a 3D DVF with zero values for the third direction
    Setting['DistanceDeform'] = 40      # The minimum distance between two random peaks
    Setting['DistanceArea'] = 20        # The area that is inculeded in the training algorithm
    Setting['sigmaNL'] = 1              # For adding noise for the next fixed image. This noise should be small otherwise we would ruin the SNR.
    Setting['Border'] = 33              # No peak would be in range of [0,Border) and [ImSize-Border, ImSize)
    Setting['sigmaN'] = 5               # Sigma for adding noise after deformation
    Setting['MaxDeform'] = [20, 15, 15] # The maximum amplitude of deformations
    Setting['sigmaB'] = [35, 25, 20]    # For blurring deformaion peak
    Setting['Np'] = [100, 100, 100]     # Number of random peaks

    # %%------------------------------------------------ Setting of reading DVFs ------------------------------------------------
    Setting['Resolution'] = 'multi'         # 'single' or 'multi' resolution. In multiresolution, the downsampled patch is involved.
    Setting['deformMethod'] = [0, 1, 2]     # 0: low freq, 1: medium freq, 2: high freq.
    Setting['classBalanced'] = [1.5, 4, 8]  # Use these threshold values to balance the number of data in each category. for instance [a,b] implies classes [0,a), [a,b). Numbers are in mm
    Setting['K'] = 65                       # Margin from the border to select random patches
    Setting['ParallelSearch'] = True        # Using np.where in parallel with [number of cores - 2] in order to make balanced data. This is done with joblib library
    Setting['R'] = 14                       # Radius of normal resolution patch size. Total size is (2*R +1)
    Setting['Rlow'] = 26                    # Radius of low resolution patch size. Total size is (Rlow +1). Selected patch size: center-Rlow : center+Rlow : 2
    Setting['Ry'] = 0                       # Radius of output. Total size is (2*Ry +1)
    Setting['verbose'] = True               # Detailed printing

    # training
    INTrain = np.arange(1, 11)          # Patients in the training set including fixed and moving images and several synthetic deformations of them
    numberOfImagesPerChunk = 5          # Number of images that I would like to load in RAM
    samplesPerImage = 10000
    batchSizeTrain = 50

    # validation
    INValidation = np.arange(11, 13)    # Patients in the validation set including fixed and moving images and several synthetic deformations of them
    numberOfImagesPerChunkVal = 2       # Number of images that I would like to load in RAM
    samplesPerImageVal = 1000
    batchSizeVal = 1000

    RegNetVal = RegNet.Patches(setting=Setting, numberOfImagesPerChunk=numberOfImagesPerChunkVal, samplesPerImage=samplesPerImageVal,  IN=INValidation, training=0)
    RegNetVal.fillList()
    RegNetTrainThread = RegNetThread.PatchesThread(setting=Setting, numberOfImagesPerChunk=numberOfImagesPerChunk, samplesPerImage=samplesPerImage, IN=INTrain, training=1)
    RegNetTrainThread.start()
    while (not RegNetTrainThread._filled):
        time.sleep(2)
    RegNetTrain = RegNet.Patches(setting=Setting, numberOfImagesPerChunk=numberOfImagesPerChunk, samplesPerImage=samplesPerImage, IN=INTrain, training=1)
    RegNetTrain.copyFromThread(RegNetTrainThread)
    chunks_completed = False
    RegNetTrainThread._filled = 0
    ThreadIsFilling = False
    mseTrainAverage = 0
    count = 1

    for itr in range(0,max_steps):
        if RegNetTrainThread._filled:
            ThreadIsFilling = False
        if (chunks_completed):
            if not RegNetTrainThread._filled:
                print ('Training the network is faster than reading the data ..... please wait .....')
                while (not RegNetTrainThread._filled):
                    time.sleep(2)
            else:
                print('Training the network is slower than reading the data  :-) ')
            RegNetTrain = RegNet.Patches(setting=Setting, numberOfImagesPerChunk=numberOfImagesPerChunk, samplesPerImage=samplesPerImage, IN=INTrain, training=1)
            RegNetTrain.copyFromThread(RegNetTrainThread)
            RegNetTrainThread._filled = 0
            chunks_completed = False
            ThreadIsFilling = False

        if (not RegNetTrainThread._filled) and (not ThreadIsFilling):
            RegNetTrainThread.goToNextChunk()
            RegNetTrainThread.resume()
            ThreadIsFilling = True


        batchX, batchY , batchXLow= RegNetTrain.next_batch(batchSizeTrain)
        if RegNetTrain._chunks_completed:
            print('chunk is completed')
            chunks_completed = True
        batchX = (batchX + 1000) / 4095.
        batchXLow = (batchXLow + 1000) / 4095.
        [mseTrainSample, _] = sess.run([mse,train_step], feed_dict={x: batchX, y: batchY, xLow: batchXLow, bn_training: 1, mseTrainAverage_net : mseTrainAverage})
        mseTrainAverage = mseTrainAverage + mseTrainSample
        count = count + 1
        if itr % 100 == 1:
            mseTrainAverage = mseTrainAverage/count
            [train_mse, s, y_dirX_temp, yHat_dirX_temp] = sess.run([mse, summ, y, yHat], feed_dict={x: batchX, y: batchY, xLow: batchXLow, bn_training: 0 ,  mseTrainAverage_net : mseTrainAverage})
            train_writer.add_summary(s, itr*batchSizeTrain)
            print('Train MSE at Epoch %s itr %s : %s  time%s' % (RegNetTrain._semiEpoch, itr*batchSizeTrain, mseTrainAverage, 2))
            mseTrainAverage = 0 ; count = 1
            if itr % 1000 == 1:
                hi = 1
                # plt.figure(figsize=(22, 12))
                # y_dirX_temp = y_dirX_temp.flatten()
                # yHat_dirX_temp = yHat_dirX_temp.flatten()
                # sort_indices = np.argsort(y_dirX_temp)
                # plt.plot(y_dirX_temp[sort_indices], label='out train dir' + str(itr*batchSizeTrain))
                # plt.plot(yHat_dirX_temp[sort_indices], label='targets dir' + str(itr*batchSizeTrain))
                # plt.legend(bbox_to_anchor=(1., .8))
                # plt.ylim((-15, 15))
                # plt.draw()
                # plt.savefig(LOGDIR + 'train' + Exp + '/Model/' + 'y_train_dir' + str(itr*batchSizeTrain) + '_epoch.png')
                # plt.close()

            RegNetVal.resetValidation()
            mseValAverage = 0
            countVal = 1
        if itr % 2000 == 1:
            while not RegNetVal._chunks_completed:
                batchXTest, batchYTest , batchXLowTest = RegNetVal.next_batch(batchSizeVal)
                batchXTest = (batchXTest + 1000) / 4095.
                batchXLowTest = (batchXLowTest + 1000) / 4095.
                [mseValSample, s, y_dirX_temp, yHat_dirX_temp] = sess.run([mse, summ, y, yHat], feed_dict={x: batchXTest, y: batchYTest, xLow: batchXLowTest, bn_training: 0, mseTrainAverage_net : mseValAverage})
                mseValAverage = mseValAverage + mseValSample
                countVal = countVal + 1
            mseValAverage = mseValAverage / countVal
            [s] = sess.run([ summ], feed_dict={ x: batchXTest, y: batchYTest, xLow: batchXLowTest, bn_training: 1, mseTrainAverage_net : mseValAverage})
            test_writer.add_summary(s, itr*batchSizeTrain)

def main():
    learning_rate = 1E-3
    max_steps = np.array(1E7).astype(np.int64)
    RegNet_model(learning_rate = learning_rate , max_steps = max_steps)


if __name__ == '__main__':
    main()