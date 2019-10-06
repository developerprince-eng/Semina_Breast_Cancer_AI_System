from __future__ import absolute_import, division, print_function

import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

os.getcwd()
os.listdir(os.getcwd())

class CREATE_DATASET():
    def __init__(self):
        self.is_data = True
    
    def __read_csv__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        return data_set
    
    def __obtain_data__(self, path, number_features, number_labels):
        data_set = pd.read_csv(path, low_memory=False)
        input_x = data_set.iloc[ : , (number_labels+1):(number_features+number_labels+1)]
        ref_input_y = data_set.iloc[ : , 1:(number_labels+1)]
        label_encoder = LabelEncoder()
        temp_y = label_encoder.fit_transform(ref_input_y)
        input_y = pd.Series(temp_y)
        
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = 0)

        return input_x, x_train, x_test, y_train, y_test

    def __obtain_labels__(self, path, number_features, number_labels):
        data_set = pd.read_csv(path, low_memory=False)
        input_x = data_set.iloc[ : , number_labels:(number_features+number_labels)]
        ref_input_y = data_set.iloc[ : , 1:(number_labels+1)]
        label_encoder = LabelEncoder()
        input_y = label_encoder.fit_transform(ref_input_y)
        
        return input_y