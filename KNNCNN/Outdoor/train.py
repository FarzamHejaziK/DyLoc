from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import load
import tensorflow as tf
#import keras.backend.tensorflow_backend as tfback
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
import statistics



data_path1 ='Data/DCNN-train.npz'
data1 = load(data_path1)
data = data1
xnum = 18
ynum = 55
num_classes = xnum * ynum
def get_data(data,xnum,ynum):
    num_classes = xnum*ynum
    #get train data
    M = np.reshape(data['ADP'],(199100,64,64,1))
    loc = data['Loc']
    x = loc[:,0]
    y = loc[:,1]
    cnew = np.zeros((len(loc),num_classes))
    c = np.zeros(len(loc))
    xnew = np.zeros(len(x))
    ynew = np.zeros(len(y))
    xmax = max(x)
    xmin = min(x)
    xstep = (xmax-xmin)/xnum
    ymax = max(y)
    ymin = min(y)
    ystep = (ymax-ymin)/ynum

    #This is for x
    #print('x: ',x[1:10])
    for i in range(len(x)):
        for j in range(1,xnum+1):
            low = xmin + (j-1)*xstep
            high = xmin + j * xstep
            if(x[i]>=low and x[i]<high):
                xnew[i] = j
            if(x[i] == xmax):
                xnew[i] = xnum
        if (xnew[i] == 0):
            print(x[i])
            print(xmax)
            print(xmin)
            sys.exit()

    #print('xnew:',xnew[1:10])
    #This is for y
    #print('y: ',y[1:10])
    for i in range(len(y)):
        for j in range(1,ynum+1):
            low = ymin + (j-1)*ystep
            high = ymin + (j*ystep)
            if(y[i]>=low and y[i]<high):
                ynew[i] = j
            if(y[i] == ymax):
                ynew[i] = ynum
        if (ynew[i] == 0):
            print(y[i])
            print(ymax)
            print(ymin)
            sys.exit()
    #print('ynew: ',ynew[1:10])
    #This creates a class
    for i in range(len(loc)):
        c[i] = (xnew[i]-1)+xnum*(ynew[i]-1)
    c = np.reshape(c,(199100,1))
    
    for i in range(len(c)):
      cnew[i,int(c[i])] = 1  
    print('cnew shape: ',cnew.shape)
   # cnew = cnew*0.01 #class is in range 0 to 1

    train_ADP, test_ADP, train_Loc, test_Loc = train_test_split(M, cnew, test_size=0.1, random_state=42)

    return loc,c,train_ADP, test_ADP, train_Loc, test_Loc
loc, c, xtrain,xtest,ytrain,ytest = get_data(data,xnum,ynum)


from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPool2D
model = Sequential()
# L = 1
model.add(Conv2D(8, (32,32), padding='same', kernel_initializer='normal',input_shape = (64,64,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(16, (16,16), padding='same', kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

# The Hidden Layers :
model.add(Conv2D(32, (8,8), padding='same', kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (7,7), padding='same', kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5,5), padding='same', kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
#conver feature maps into single layer
#conver feature maps into single layer

#model.add(Conv2D(256, (5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, (5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten()) 
#fully connected layer
#input of FCL
model.add(Dense(256, activation="relu"))
#output of FCL
model.add(Dense(num_classes, activation = 'softmax')) #output shape
#Trail 1
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#Trail 2

model.summary()

#checkpoint_name = '/content/drive/My Drive/My Code Cleaned/data/paperO1/KNNCNNweightsnew1/Weights6-{epoch:03d}--{val_loss:.5f}.hdf5'
#checkpoint = ModelCheckpoiant(checkpoint_name, monitor='val_accuracy', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]
#model.fit(xtrain, ytrain, batch_size=100,epochs=1000, verbose=1, validation_data = (xtest,ytest))
#ypred = model.predict(xtest)
#loss, acc =model.evaluate(xtest, ytest,verbose = 0) 
#print('loss: ',loss)
#print('accuracy: ',acc)

checkpoint_name = 'KNNCNNW1/Weights-{epoch:03d}--{val_accuracy:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
model.fit(xtrain, ytrain, batch_size=80,epochs=10000, verbose=1, validation_data = (xtest,ytest), callbacks = callbacks_list)