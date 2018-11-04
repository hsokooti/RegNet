import os
import numpy as np
import matplotlib.pyplot as plt
import functions.setting_utils as su


def regressionPlot(y_plot, yHat_plot, itr, batchSizeTrain, plot_mode, setting=None):
    if not (os.path.isdir(su.address_generator(setting, 'Plots_folder'))):
        os.makedirs(su.address_generator(setting, 'Plots_folder'))

    y_dir_plot = np.empty([(np.shape(y_plot[:, :, :, :, 0].flatten()))[0], 3])
    yHat_dir_plot = np.empty([(np.shape(yHat_plot[:, :, :, :, 0].flatten()))[0], 3])

    for i in range(3):
        try:
            y_dir_plot[:, i] = y_plot[:, :, :, :, i].flatten()
            yHat_dir_plot[:, i] = yHat_plot[:, :, :, :, i].flatten()
            plt.figure(figsize=(22, 12))
            sort_indices = np.argsort(y_dir_plot[:, i])
            plt.plot(yHat_dir_plot[:, i][sort_indices], label='RegNet dir' + str(i) + '_itr' + str(itr * batchSizeTrain))
            plt.plot(y_dir_plot[:, i][sort_indices], label='y dir' + str(i) + '_itr' + str(itr * batchSizeTrain))
            plt.legend(bbox_to_anchor=(1., .8))
            plt.ylim((-22, 22))
            plt.draw()
            plt.savefig(su.address_generator(setting, 'plot_fig', plot_mode=plot_mode, plot_itr=itr * batchSizeTrain, plot_i=i))
            plt.close()
        except:
            print('error in plotting... ')
            pass


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')


# class AllPoints(object):
#     def __init__(self, AllPoints, INTest, negative=False):
#         self._AllPoints = AllPoints
#         self._INTest = INTest
#         self._negative = negative
#         self._AllPointsMerged = self.mergeAllPoints()
#         self._TRE = {}
#         self._Err = {}
#         self._DVF = {}
#         self.TRE_initialize()   # calculate affine TRE
#
#     def mergeAllPoints(self):
#         AllPointsMerged = {}
#         allPointsfileds = ['FixedLandmarksWorld', 'MovingLandmarksWorld', 'FixedAfterAffineLandmarksWorld', 'DVFAffine', 'DVFRegNet', 'DVF_nonrigidGroundTruth']
#         for i, IN in enumerate(self._INTest):
#             for field in allPointsfileds:
#                 if i == 0:
#                     AllPointsMerged[field] = self._AllPoints[IN][field]
#                 else:
#                     AllPointsMerged[field] = np.concatenate((AllPointsMerged[field], self._AllPoints[IN][field]), axis=0)
#         if self._negative:
#             AllPointsMerged['DVFRegNet'] = -AllPointsMerged['DVFRegNet']
#         return AllPointsMerged
#
#     def TRE_initialize(self):
#         TRE_affine_out = [np.linalg.norm(self._AllPointsMerged['MovingLandmarksWorld'][i, :] - self._AllPointsMerged['FixedAfterAffineLandmarksWorld'][i, :])
#                         for i in range(np.shape(self._AllPointsMerged['MovingLandmarksWorld'])[0])]
#         Err_affine_out = [(self._AllPointsMerged['MovingLandmarksWorld'][i, :] - self._AllPointsMerged['FixedAfterAffineLandmarksWorld'][i, :])
#                         for i in range(np.shape(self._AllPointsMerged['MovingLandmarksWorld'])[0])]
#         TRE_groundTruth = [np.linalg.norm(self._AllPointsMerged['DVF_nonrigidGroundTruth'][i, :]) for i in range(np.shape(self._AllPointsMerged['MovingLandmarksWorld'])[0])]
#
#         self._TRE['Affine'] = np.array(TRE_affine_out, dtype=np.float32)
#         self._TRE['GroundTruth'] = np.array(TRE_groundTruth, dtype=np.float32)
#         self._Err['Affine'] = np.array(Err_affine_out, dtype=np.float32)
#         self._DVF['GroundTruth'] = self._AllPointsMerged['DVF_nonrigidGroundTruth']
#
#     def TRE_calculateFromErr(self, Error, methodName):
#         # Error is Affine - DVF
#         TRE = [np.linalg.norm(Error[i,:])
#                         for i in range(np.shape(Error)[0])]
#         self._TRE[methodName] = np.array(TRE, dtype=np.float32)
#         self._Err[methodName] = np.array(Error, dtype=np.float32)
#
#     def TRE_calculateFromDVF(self, DVF, methodName):
#         TRE_DVF_out = [np.linalg.norm(self._AllPointsMerged['MovingLandmarksWorld'][i, :] -
#                                       self._AllPointsMerged['FixedAfterAffineLandmarksWorld'][i, :] -
#                                       DVF[i, :])
#                        for i in range(np.shape(self._AllPointsMerged['MovingLandmarksWorld'])[0])]
#         Err_DVF_out = [self._AllPointsMerged['MovingLandmarksWorld'][i, :] -
#                        self._AllPointsMerged['FixedAfterAffineLandmarksWorld'][i, :] -
#                        DVF[i, :]
#                        for i in range(np.shape(self._AllPointsMerged['MovingLandmarksWorld'])[0])]
#         self._TRE[methodName] = np.array(TRE_DVF_out, dtype=np.float32)
#         self._Err[methodName] = np.array(Err_DVF_out, dtype=np.float32)
#         self._DVF[methodName] = np.array(DVF, dtype=np.float32)
#
#
# def boxTREFull(TRE, exp_list, normalize=True, min1=20, max1=90, min2=20, max2=30, dashLine=None):
#     TRE = copy.deepcopy(TRE)
#     if normalize:
#         for exp in exp_list:
#             Ix = np.where(TRE[exp] > min1)
#             TRE[exp][Ix] = (TRE[exp][Ix] - min1) * (max2 - min1) / (max1 - min2) + min2
#
#     plt.figure(figsize=(8, 6))
#     outPlot = plt.boxplot([TRE[exp] for exp in exp_list])
#     plt.xticks(np.arange(1, len(exp_list) + 1), [exp for exp in exp_list], fontsize=16, rotation=90)
#     plt.yticks([0, 5, 10, 15, min1, max2], ['0', '5', '10', '15', str(min1), str(max1)], fontsize=24)
#     if dashLine is not None:
#         I = np.where(np.array(exp_list)==dashLine)
#         plt.axhline(y=outPlot['boxes'][I[0][0]]._path.vertices[2,1], color='r', linestyle=':')
#
#     plt.subplots_adjust(hspace=0, bottom=0.5)
#     plt.title('TRE [mm] of all landmarks')
#     plt.rc('font', family='serif')
#     plt.draw()
#
#
# def boxTRECrop(TRE, exp_list, threshold, AffineError, normalize=True, dashLine=None):
#     selectedLandmarks = np.where(np.all(np.abs(AffineError) <= threshold, axis = 1))
#     TRECrop = {}
#     for exp in exp_list:
#         TRECrop[exp] = TRE[exp][selectedLandmarks]
#     plt.figure(figsize=(8, 6))
#     outPlot = plt.boxplot([TRECrop[exp] for exp in exp_list])
#     plt.xticks(np.arange(1, len(exp_list) + 1), [exp for exp in exp_list], fontsize=16, rotation=90)
#     yTicksNum = [0, 5, 10, 15]
#     yTickStr = ['0', '5', '10', '15']
#     while yTicksNum[-1] < threshold+ 5:
#         yTicksNum.append(yTicksNum[-1]+5)
#         yTickStr.append(str(yTicksNum[-2] + 5))
#     plt.yticks(yTicksNum, yTickStr, fontsize=24)
#     if dashLine is not None:
#         I = np.where(np.array(exp_list)==dashLine)
#         plt.axhline(y=outPlot['boxes'][I[0][0]]._path.vertices[2,1], color='r', linestyle=':')
#
#     plt.subplots_adjust(hspace=0, bottom=0.5)
#     plt.title('TRE [mm] of the capture range of {}'.format(threshold))
#     plt.rc('font', family='serif')
#     plt.draw()
#
#
# def tableTRE(TRE, Err, exp_list, threshold=None):
#     AffineError = Err['Affine']
#     tab = {}
#     latexStr = {}
#     headerTab = ['exp', 'measure', 'measure_x', 'measure_y', 'measure_z']
#     valueTabs = np.empty([len(exp_list), 5], dtype=object)
#     if threshold is not None:
#         selectedLandmarks = np.where(np.all(np.abs(AffineError) <= threshold, axis=1))
#         selectedLandmarks = selectedLandmarks[0]
#     else:
#         selectedLandmarks = np.arange(0, len(AffineError))
#
#     for i, exp in enumerate(exp_list):
#         tab[exp] = {}
#         latexStr[exp] = exp + ' '
#         tab[exp]['mean'] = np.mean(TRE[exp][selectedLandmarks])
#         tab[exp]['std'] = np.std(TRE[exp][selectedLandmarks])
#         valueTabs[i, 0] = exp
#         valueTabs[i, 1] = '${:.2f}\pm{:.2f}$'.format(tab[exp]['mean'], tab[exp]['std'])
#         latexStr[exp] += '${:.2f}\pm{:.2f}$'.format(tab[exp]['mean'], tab[exp]['std'])
#         for dim in range(3):
#             tab[exp]['dim' + str(dim)] = {}
#             tab[exp]['dim' + str(dim)]['mean'] = np.mean(np.abs([Err[exp][k][dim] for k in selectedLandmarks]))
#             tab[exp]['dim' + str(dim)]['std'] = np.std(np.abs([Err[exp][k][dim] for k in selectedLandmarks]))
#             valueTabs[i, dim+2] = '${:.2f}\pm{:.2f}$'.format(tab[exp]['dim' + str(dim)]['mean'], tab[exp]['dim' + str(dim)]['std'])
#             latexStr[exp] += ' & ' + '${:.2f}\pm{:.2f}$'.format(tab[exp]['dim' + str(dim)]['mean'], tab[exp]['dim' + str(dim)]['std'])
#         print (latexStr[exp])
#
#     fig, ax = plt.subplots(figsize=(15,8))
#     fig.patch.set_visible(False)
#     ax.table(cellText=valueTabs, colLabels=headerTab, loc='center', colLoc='left', cellLoc='left', fontsize=50)
#     ax.axis('off')
#     ax.axis('tight')
#     fig.tight_layout()
#     if threshold:
#         plt.title('TRE [mm] of the capture range of {}'.format(threshold))
#     plt.rc('font', family='serif')
#     plt.draw()
#
# # A='${:.2f}\pm{:.2f}$  & ${:.2f}\pm{:.2f}$  & ${:.2f}\pm{:.2f}$  & ${:.2f}\pm{:.2f}$'.format(TREBspline1res_mean,TREBspline1res_std,\
# # MAEBspline1res_dir_mean[0],MAEBspline1res_dir_std[0],MAEBspline1res_dir_mean[1],MAEBspline1res_dir_std[1],MAEBspline1res_dir_mean[2],MAEBspline1res_dir_std[2])
# #
#
#
# def scatterTREFull(TRE, myMethod, normalize=True, min1=25, max1=90, min2=25, max2=40):
#     TRE = copy.deepcopy(TRE)
#     exp_list = ['GroundTruth', myMethod]
#     if normalize:
#         for exp in exp_list:
#             Ix = np.where(TRE[exp] > min1)
#             TRE[exp][Ix] = (TRE[exp][Ix] - min1) * (max2 - min1) / (max1 - min2) + min2
#     plt.figure(figsize=(12, 6))
#     plt.plot(TRE['GroundTruth'], TRE[myMethod], 'o', color='cyan')
#     xTickPoints = np.arange(0, min1+5, 5)
#     xTickLabels = [str(myNum) for myNum in xTickPoints]
#     xTickPoints = np.append(xTickPoints, max2)
#     xTickLabels.append(str(max1))
#     yTickPoints = np.arange(0, min1+5, 5)
#     plt.xticks(xTickPoints, xTickLabels, fontsize=20, rotation=30)
#     plt.yticks(yTickPoints, [str(myNum) for myNum in yTickPoints], fontsize=20, rotation=90)
#     plt.plot([0, min1],[0, min2], color='blue')
#     plt.ylim([0, max(TRE['GroundTruth'])])
#     plt.xlim([0, xTickPoints[-1]])
#     plt.xlabel('Ground truth [mm]', fontsize=20)
#     plt.ylabel(myMethod + ' [mm]', fontsize=20)
#     plt.rc('font', family='serif')
#     plt.subplots_adjust(hspace=0, bottom=0.2)
#     plt.draw()
#
#
# def scatterDVF(DVF, Err, myMethod, threshold=0):
#     AffineError = Err['Affine']
#     if threshold:
#         selectedLandmarks = np.where(np.all(np.abs(AffineError) <= threshold, axis = 1))
#     else:
#         selectedLandmarks = np.arange(0, len(AffineError))
#     DVFCrop = {}
#     exp_list = ['GroundTruth', myMethod]
#     for exp in exp_list:
#         DVFCrop[exp] = DVF[exp][selectedLandmarks]
#     for dim in range (3):
#         plt.figure(figsize=(6, 6))
#         plt.scatter(DVFCrop['GroundTruth'][:,dim], DVFCrop[myMethod][:,dim], color='green')
#         plt.ylim((-threshold-5,threshold+5))
#         plt.xlim((-threshold-5,threshold+5))
#         plt.title('Dimension'+str(dim))
#         plt.xlabel('Ground truth [mm]', fontsize=20)
#         plt.ylabel(myMethod + ' [mm]', fontsize=20)
#         plt.plot([-threshold, threshold], [-threshold, threshold], color='blue', linewidth=2)
#         plt.draw()

