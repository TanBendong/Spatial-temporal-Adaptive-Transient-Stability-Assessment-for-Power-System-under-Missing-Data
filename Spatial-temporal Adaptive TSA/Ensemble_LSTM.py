# ----------------------------------------------------
# Description: Class for DynamicLSTM Network
# Created by: Bendong Tan
# Created time: Thursday, Feb 14, 2019
# Last Modified: Thursday, Feb 14, 2019
# Wuhan University
# ----------------------------------------------------
import scipy.io as sio
import numpy as np
import time
from Mail import Mail
from DynamicLSTM import DynamicLSTM
import os
import shutil
from keras import backend as K
import tensorflow as tf

# calculate running time
start = time.time()
file=''
validation_mat=np.zeros((17,10))
# load data
for T in range(1,11,1):
    # create model
    for i in range(17):
        # load data
        X_trains = sio.loadmat(file + '\X_train' + str(i+1) + '.mat')  # training dataset
        X_train = X_trains['x_train']

        X_validations = sio.loadmat(file + '\X_validation' + str(i+1) + '.mat')  # validation dataset
        X_validation = X_validations['x_validation']

        y_trains = sio.loadmat(file + '\y_train.mat')  # training labels
        y_train = y_trains['y_train']

        y_validations = sio.loadmat(file + '\y_validation.mat')  # validation labels
        y_validation = y_validations['y_test']

        Adaptive_data = X_validation
        ACC = 0
        X_train, y_train, X_validation, y_validation = X_train[:, :T, :], y_train, X_validation[:, :T,
                                                                                       :], y_validation
        LSTM = DynamicLSTM(X_train, y_train, X_validation, y_validation)
        model = LSTM.build()

        model.load_weights(r".\\"+str(T)+"\model_best" + str(T) + "_" + str(i+1) + ".h5")


        delta = 0.5

        ART, Accuracy = LSTM.Adaptive_TSA(model, Adaptive_data, y_validation, delta)
        prediction = model.predict(X_validation)
        for pred_i in prediction:
            pred_i[pred_i >= 0.5] = 1
            pred_i[pred_i < 0.5] = 0
        miss = 0
        for j in range(len(y_validation)):
            if prediction[j] != y_validation[j]:
                miss = miss + 1
        validation_mat[i,T-1]=1/(1-(len(y_validation) - miss) / len(y_validation))

        print("ART:" + str(ART) + " 秒")
        print("accuracy:" + str((len(y_validation) - miss) / len(y_validation) * 100) + " %")
        print("Delta accuracy:" + str(Accuracy) + " %")
        K.clear_session()
        tf.reset_default_graph()
sio.savemat("validation.mat", {'validation': validation_mat})
end=time.time()
print(end-start)