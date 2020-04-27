# ----------------------------------------------------
# Description: Class for DynamicLSTM Network
# Created by: Bendong Tan
# Created time: Friday, Jan 25, 2019
# Last Modified: Monday, Jan 30, 2019
# Wuhan University
# ----------------------------------------------------
import os
os.environ['KERAS_BACKEND']='tensorflow'
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Masking,Flatten,Dropout,LSTM,Average
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import regularizers


class DynamicLSTM:
    '''
    初始化
    '''
    def __init__(self, X_train, y_train, X_test, y_test):
        n_trainsample, n_timesteps, n_features = np.shape(X_train)
        # training dataset
        self.X_train = X_train
        # testing dataset
        self.X_test = X_test
        # training label
        self.y_train = y_train
        # testing label
        self.y_test = y_test
        # final output
        self.n_outputs = 1
        # feature dimension at each instant
        self.n_features = n_features
        # the length of time
        self.n_timesteps = n_timesteps
        # layer number
        self.layer_num = 1
        # the number of hidden nueral layer
        self.hidden_size=100
        # batch size
        self.batch_size=n_trainsample
        # learning rate
        self.learningRate = 1e-3
        # training epochs
        self.epochs=200
        # file to save model
        self.storePath = None
    '''
    build LSTM网络
    '''
    def build(self):
        # begin to build the model
        model = Sequential()
        # zero padding for time-adaptive TSA to keep the leanth of the input
        model.add(Masking(mask_value=0, input_shape=(self.n_timesteps, self.n_features)))
        # LSTM layer
        model.add(LSTM(self.hidden_size,return_sequences=True))
        model.add(Dropout(0.05))
        model.add(LSTM(self.hidden_size, return_sequences=True))
        model.add(Dropout(0.05))
        model.add(LSTM(self.hidden_size))
        # sigmoid layer
        model.add(Dense(self.n_outputs, activation='sigmoid'))
        # compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        return model
    '''
    train the model
    '''
    def fit(self,model,T=1,L=1):
        # record the best model
        filepath = r".\model\model_best"+str(T)+"_"+str(L)+".h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        # train the model
        model.fit(self.X_train, self.y_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(self.X_test, self.y_test),
                  callbacks=callbacks_list,
                  verbose=1)

        return model
    '''
    assessment 
    '''
    def evaluation(self,model,Dictionary, y_test,validation):
        predictions=np.zeros((len(y_test),20))
        for t in range(20):  # max length of assessment time
            print('the assessment instant is:' + str(t))
            if t < self.n_timesteps:
                for k in range(len(model)):
                    X_test = Dictionary[str(k + 1)]
                    Input = pad_sequences(X_test[:, 0:t + 1, :],maxlen=self.n_timesteps, padding='post')
                    predictions[:,t:t+1] = predictions[:,t:t+1] + model[k].predict(Input) * validation[k] / sum(validation)
            if t >= self.n_timesteps:
                for k in range(len(model)):
                    X_test = Dictionary[str(k + 1)]
                    Input = X_test[:, t-self.n_timesteps+1:t + 1, :]
                    predictions[:,t:t+1] = predictions[:,t:t+1] + model[k].predict(Input) * validation[k] / sum(validation)
        return predictions
    '''
    time-adaptive assessment model
    '''
    def Adaptive_TSA(self,prediction,y_test,delta):
        miss = 0 # record the number of misclassification
        right = np.zeros((len(y_test), 20)) # record the number of classification
        y_pred = np.zeros((len(y_test), 1)) # final TSA results

        # the process of time-adaptive assessment
        for i in range(len(y_test)):
            for t in range(20): 
                predictions=prediction[i,t]
                if predictions >= delta and predictions <= 1:
                    right[i, 0:t + 1] = 1
                    y_pred[i] = 1
                    break
                if predictions >=0  and predictions < 1-delta:
                    right[i, 0:t + 1] = 1
                    y_pred[i] = 0
                    break
                # samples is seen as unstable when the assessment time exceed the max time
                if t + 1 == 20:
                    if predictions >= 0.5:
                        y_pred[i] = 1
                    if predictions < 0.5:
                        y_pred[i] = 0
                    right[i, 0:t + 1] = np.ones((1, t + 1))
                    break
        # calculate accuracy
        for i in range(len(y_test)):
            if y_pred[i]!=y_test[i]:
                miss = miss + 1

        # record average response time
        ART = sum(sum(right)) / len(y_test)

        # record accuracy
        Accuracy=(len(y_test)-miss)/len(y_test)*100

        return ART, Accuracy
