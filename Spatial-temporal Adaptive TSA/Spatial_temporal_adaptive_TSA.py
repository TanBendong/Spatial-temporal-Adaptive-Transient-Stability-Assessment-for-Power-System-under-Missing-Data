# ----------------------------------------------------
# Description: Spatial-temporal adaptive Transient Stability Assessment for Power System under Missing Data
# Created by: Bendong Tan
# Created time: Thu, Feb 20, 2019
# Last Modified: Thu, Feb 20, 2019
# Wuhan University
# ----------------------------------------------------
import scipy.io as sio
import numpy as np
import time
from Ensemble_DynamicLSTM import DynamicLSTM
import os
from keras import backend as K
import tensorflow as tf
import itertools
import os
from multiprocessing import Pool
from functools import partial


def main(LSTM,Dictionary,t, MODEL,validation,Vector,miss):
    T = 4
    y_trains = sio.loadmat('y_train.mat')  # training labels
    y_train = y_trains['y_train']

    y_tests = sio.loadmat('y_test.mat')  # testing label
    y_test = y_tests['y_test']
    for i in range(17):
        # load datasets
        X_tests = sio.loadmat('X_test' + str(i + 1) + '.mat')  # testing datasets
        X_test = X_tests['x_test']
        if len(set(Vector[str(i + 1)]) & set(miss)) > 0:
            X_test[:, t:, :] = 0

        Dictionary[str(i + 1)] = X_test

    predictions = LSTM.evaluation(MODEL, Dictionary, y_test, validation[:, T - 1])
    ART, Accuracy = LSTM.Adaptive_TSA(predictions, y_test, 0.55)

    return ART, Accuracy
def test(t=0,n=1):
    # calculate running time of our computer
    start = time.time()
    PMU_clusters=np.loadtxt('17.csv',delimiter=',')
    Vector={}
    for k in range(17):
        vector=np.where(PMU_clusters[k,:])
        V=np.array(vector) + 1
        Vector[str(k+1)] = V[0]
    # creat model
    MODEL = []  # save ensemble model
    Dictionary = {}
    y_trains = sio.loadmat('y_train.mat')  # training labels
    y_train = y_trains['y_train']

    y_tests = sio.loadmat('y_test.mat')  # testing labels
    y_test = y_tests['y_test']

    validations = sio.loadmat('validation.mat')
    validation = validations['validation']
    for i in range(17):
        # load data

        X_trains = sio.loadmat('X_train' + str(i + 1) + '.mat')  # training data
        X_train = X_trains['x_train']

        X_tests = sio.loadmat('X_test' + str(i + 1) + '.mat')  # testing data
        X_test = X_tests['x_test']

        Dictionary[str(i + 1)] = X_test
        X_train, y_train, X_test, y_test = X_train[:, :4, :], y_train, X_test[:, :4, :], y_test

        LSTM = DynamicLSTM(X_train, y_train, X_test, y_test)
        model = LSTM.build()

        model.load_weights("model_best" + str(4) + "_" + str(i + 1) + ".h5")
        MODEL.append(model)
    # Spatial-temporal adaptive TSA under Missing Data
    MISS = []
    for miss in itertools.product([1,2,3,4,5,6,7,8], repeat=n):
        # print(np.array(miss))
        MISS.append(np.array(miss))
    pool = Pool()
    func = partial(main, LSTM,Dictionary,1,MODEL,validation,Vector)
    ART=pool.map(func, MISS)
    pool.close()
    pool.join()
    end = time.time()

    return ART
if __name__ == "__main__":
    start = time.time()
    ART_mat = np.zeros((20, 7))
    ACC_mat = np.zeros((20, 7))
    for t in range(20):
        print('##################'+'时刻数:'+str(t)+'##################')
        for n in range(1,8,1):
            A=test(t=t,n=n)
            B = np.array(A)
            C=np.sum(B, 0)
            ART_mat[t, n - 1] = C[0]
            ACC_mat[t, n - 1] = C[1]
    sio.savemat('ART_mat.mat', {'ART_mat': ART_mat})
    sio.savemat('ACC_mat.mat', {'ACC_mat': ACC_mat})
    end = time.time()