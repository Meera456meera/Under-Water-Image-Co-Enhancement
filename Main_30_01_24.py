import os
import cv2 as cv
import numpy as np
from numpy import matlib
import random as rn
from BWO import BWO
from CO import CO
from FCM import FCM
from Glob_Vars import Glob_Vars
from KOA import KOA
from Model_CNN import Model_CNN
from Model_Transunet import Model_TransUnet
from Model_Unet import Mode_Unet
from Model_Unet3plus import Model_Unet3plus
from Objective_Function import Objective_Seg
from PROPOSED import PROPOSED
from Pred_PlotResults import plot_results_Pred, plot_results_conv
from TSO import TSO

# Read Dataset
an = 0
if an == 1:
    Dir = './Dataset/'
    list_dir = os.listdir(Dir)
    original = []
    co_enhancement = []
    for i in range(len(list_dir)):
        file = Dir + list_dir[i] +'/'
        list_dir1 = os.listdir(file)
        for j in range(len(list_dir1)):
            if '.tif' in list_dir1[j]:
                if 'resized' in list_dir1[j]:
                    pass
                else:
                    file1 = file + list_dir1[j]
                    read = cv.imread(file1)
                    read = cv.resize(read,[512,512])
                    original.append(read)
    np.save('Original_Image.npy',np.asarray(original))

## Generate GroundTruth
an = 0
if an == 1:
    GT = []
    Images = np.load('Original_Image.npy', allow_pickle=True)
    for k in range(len(Images)):
        print('Image', k)
        img = Images[k]
        imag = cv.resize(img, [512, 512])
        image = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)
        cluster = FCM(image, image_bit=8, n_clusters=8, m=10, epsilon=0.8, max_iter=30)
        cluster.form_clusters()
        result = cluster.result.astype('uint8') * 30
        values, counts = np.unique(result, return_counts=True)
        index = np.argsort(counts)[::-1][2]
        result[result != values[index]] = 0
        analysis = cv.connectedComponentsWithStats(result, 4, cv.CV_32S)
        (totalLabels, Img, values, centroid) = analysis
        uniq, counts = np.unique(Img, return_counts=True)
        zeroIndex = np.where(uniq == 0)[0][0]
        uniq = np.delete(uniq, zeroIndex)
        counts = np.delete(counts, zeroIndex)
        sortIndex = np.argsort(counts)[::-1]
        uniq = uniq[sortIndex]
        counts = counts[sortIndex]
        Img = Img.astype('uint8')
        remArray = []
        for j in range(len(counts)):
            if counts[j] < 100 or counts[j] > 750:
                Img[Img == uniq[j]] = 0
                remArray.append(j)
        if not remArray:
            pass
        else:
            remArray = np.array(remArray)
            uniq = np.delete(uniq, remArray)
            counts = np.delete(counts, remArray)
            Img[Img != 0] = 255
            Img = Img.astype('uint8')
            kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(Img, cv.MORPH_OPEN, kernel, iterations=1)
            kernel = np.ones((3, 3), np.uint8)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1).astype('uint8')
            GT.append(closing)
    np.save('Ground_Truth.npy', GT)

# Optimization for Segmentation
an = 0
if an == 1:
    Images = np.load('Original_Image.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    Glob_Vars.Images = Images
    Glob_Vars.Target = GT
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(([5, 5, 300]), Npop, 1)
    xmax = matlib.repmat(([255, 50, 1000]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Seg
    max_iter = 50

    print('TSO....')
    [bestfit1, fitness1, bestsol1, Time1] = TSO(initsol, fname, xmin, xmax, max_iter)

    print('BWO....')
    [bestfit2, fitness2, bestsol2, Time2] = BWO(initsol, fname, xmin, xmax, max_iter)

    print('CO....')
    [bestfit3, fitness3, bestsol3, Time3] = CO(initsol, fname, xmin, xmax, max_iter)

    print('KOA....')
    [bestfit4, fitness4, bestsol4, Time4] = KOA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol=([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    fitness=([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)

# Adaptive Trans-ResUnet Segmentation
an = 0
if an == 1:
    img = np.load('Original_Image.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)[4, :]
    per = round(img.shape[0] * 0.75)
    train_data = img[:per]
    train_target = GT[:per]
    test_data = img[per:]
    test_target = GT[per:]
    Eval,Images = Model_TransUnet(train_data, train_data, test_data, test_target, sol.astype('int'))
    np.save('Co_Enhancement.npy', Images)

# Segmentation Comparison
an = 0
if an == 1:
    Eval_all = []
    Images = np.load('Original_Image.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    per = round(Images.shape[0] * 0.75)
    Eval = np.zeros((10, 3))
    train_data = Images[:per]
    train_target = GT[:per]
    test_data = Images[per:]
    test_target = GT[per:]
    for i in range(5):
        Eval[i, :],Image = Model_TransUnet(train_data, train_data, test_data, test_target, sol[i].astype('int'))
    Eval[5, :]= Model_CNN(train_data, train_data, test_data, test_target)
    Eval[6, :] = Mode_Unet(train_data, train_data, test_data, test_target)
    Eval[7, :] = Model_Unet3plus(train_data, train_data, test_data, test_target)
    Eval[8, :]= Model_TransUnet(train_data, train_data, test_data, test_target)
    Eval[9, :], Image5 = Eval[4, :]
    np.save('Eval_all.npy', Eval_all)

plot_results_Pred()
plot_results_conv()