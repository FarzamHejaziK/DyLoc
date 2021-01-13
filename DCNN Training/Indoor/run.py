from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from numpy import load
import time
import sys
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

## Data Preparation to feed the CNN training
def get_data():
    #get train data
    data_path ='Data/TrainDCNNI1.npz'
    data = load(data_path)
    M = np.reshape(data['ADP'],(550*121,32,32,1))
    train_ADP, test_ADP, train_Loc, test_Loc = train_test_split(M, data['Loc'], test_size=0.001, random_state=42)

    return train_ADP, test_ADP, train_Loc, test_Loc


train_ADP, test_ADP, train_Loc, test_Loc = get_data()


## Neural Network Artitechture and Training
with tf.device("gpu:0"):
    NN_model = Sequential()
# The Input Layer :
    NN_model.add(Conv2D(4, (16,16), padding='same', kernel_initializer='normal',input_shape = (32,32,1), activation='relu'))
    NN_model.add(BatchNormalization())
    NN_model.add(MaxPool2D(pool_size=(2, 2)))

# The Hidden Layers :
    NN_model.add(Conv2D(8, (8,8), padding='same', kernel_initializer='normal', activation='relu'))
    NN_model.add(BatchNormalization())
    NN_model.add(MaxPool2D(pool_size=(2, 2)))
    NN_model.add(Conv2D(16, (7,7), padding='same', kernel_initializer='normal', activation='relu'))
    NN_model.add(BatchNormalization())
    NN_model.add(MaxPool2D(pool_size=(2, 2)))
    NN_model.add(Conv2D(32, (5,5), padding='same', kernel_initializer='normal', activation='relu'))
    NN_model.add(BatchNormalization())
    NN_model.add(MaxPool2D(pool_size=(2, 2)))
    NN_model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='normal', activation='relu'))
    NN_model.add(BatchNormalization())
    NN_model.add(MaxPool2D(pool_size=(2, 2)))
# The Output Layer :
    NN_model.add(Flatten()) 
    NN_model.add(Dense(3))
# Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()
    checkpoint_name = 'weigths/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
    NN_model.fit(train_ADP, train_Loc, epochs=10000, batch_size=100, validation_split = 0.05, callbacks=callbacks_list)