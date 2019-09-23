from __future__ import absolute_import, division, print_function

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 

os.getcwd()
os.listdir(os.getcwd())

class GENERATE_MODEL():
    def __init__(self):
        self.is_model = True

    #Train RNN Classifier
    def kr_train_RNN_Model(self):
        metrics = []
        return metrics
    
    #Train CNN Classifier
    def kr_train_CNN_Model(self):
        metrics = []
        return metrics

    def kr_train_DNN_Seq_01(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=x_dim, init='uniform', activation='relu'))
        model.add(Dense(60, init='uniform', activation='relu'))
        model.add(Dense(30, init='uniform', activation='sigmoid'))
        model.add(Dense(2, init='uniform', activation='sigmoid'))                                                                                                                                                                                                                       
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
        # Fit the model
        model.fit(features_train, labels_train, epochs=150, batch_size=batch_size, verbose=2)
     
        score = model.evaluate(features_test, labels_test, verbose=1)
        # round predictions
        accuracy = score[1]
    
        return accuracy

    # DNN CLassifier
    def tf__evaluate_DNN__(self, input_x ,features_train ,features_test, labels_train , labels_test, batch_size):
        estimator = tf.estimator.DNNClassifier(
                    feature_columns=input_x,
                    hidden_units=[1024, 512, 256, 64],
                    optimizer=tf.train.ProximalAdagradOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=0.001
                    ))
        

        def input_fn_train(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset
            
        
        def input_fn_eval(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset


        def input_fn_predict(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset
        
        input_fn_train = input_fn_train(features_train, labels_train, batch_size)
        input_fn_eval = input_fn_eval(features_test, labels_test, batch_size)   
        input_fn_predict = input_fn_predict(features_test, labels_test, batch_size)

        estimator.train(input=input_fn_train)
        metrics = estimator.evaluate(input_fn=input_fn_eval)
        predictions = estimator.predict(input_fn=input_fn_predict)
        
        return metrics, predictions
    
