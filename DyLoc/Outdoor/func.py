# Main Database Load

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import warnings
from numpy import load
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)



# All functions Defined

def Loc_Credible(ADP,Loc,Loc_database,ADP_database,Previous_Loc):
    if len(Previous_Loc) > 0 and  np.linalg.norm(Loc - Previous_Loc) > 2 :
        out = 'Location is not Credible'
        return out
    KNN = np.where((np.linalg.norm(Loc_database - Loc, axis = 1 ) < 0.3))
    KNN = np.asarray(KNN).T
    ADP = np.reshape(ADP,(1,64,64))
    if len(KNN) == 0:
        out = 'Location is not Credible'
        return out
    for idx in KNN:
        if mse(ADP_database[idx],ADP) > 0.97 and not(np.isnan(mse(ADP_database[idx],ADP))) :
            out = 'Location is Credible'
            return out
    out = 'Location is not Credible'
    return out



def Loc_ADP_Recovery(ADP,Pred_ADP,Previous_Loc,Loc_database,ADP_database):
    KNN = np.where((np.linalg.norm(Loc_database - Previous_Loc, axis = 1 ) < 1))
    KNN = np.asarray(KNN).T
    W = []
    if (np.linalg.norm(FP_Localizer(Pred_ADP) - Previous_Loc) > 1):
        Pred_Loc = Previous_Loc
    else:
        Pred_Loc = FP_Localizer(Pred_ADP)

    if not(np.isnan(mse(ADP_database[KNN[0]],ADP))):
            W = mse(ADP_database[KNN[0]],ADP)
            KNN_Loc = Loc_database[KNN[0]]
            KNN_ADP = ADP_database[KNN[0]]

    for idx in KNN[1:len(KNN)]:
        if not(np.isnan(mse(ADP_database[idx],ADP))):
            W = np.append(W,mse(ADP_database[idx],ADP))
            KNN_Loc = np.append(KNN_Loc, Loc_database[idx], axis = 0)
            KNN_ADP = np.append(KNN_ADP, ADP_database[idx], axis = 0)

    PredADP = np.reshape(Pred_ADP,(1,64,64))
    if not(np.isnan(mse(PredADP,ADP))):
        W = np.append(W,mse(PredADP,ADP))
        KNN_Loc = np.append(KNN_Loc,np.reshape(Pred_Loc,(1,3)), axis = 0)
        KNN_ADP = np.append(KNN_ADP,PredADP, axis = 0)
    Est_Loc = np.zeros((1,3))
    Est_ADP = np.zeros((64,64))
    
    if not(len(W) == 0):
        W = W/(np.sum(W))
        
        for i in range(0,len(W)):
            Est_Loc = Est_Loc + W[i] * KNN_Loc[i]
            Est_ADP = Est_ADP + W[i] * KNN_ADP[i]
    else:
        Est_Loc = Pred_Loc
        Est_ADP = np.reshape(Pred_ADP,(64,64))
    return Est_Loc, Est_ADP

def Loc_ADP_Recovery_WO_Pred(ADP,Pred_ADP,Previous_Loc,Loc_database,ADP_database):
    KNN = np.where((np.linalg.norm(Loc_database - Previous_Loc, axis = 1 ) < 1.5))
    KNN = np.asarray(KNN).T
    W = []
    Pred_Loc = Previous_Loc

    if not(np.isnan(mse(ADP_database[KNN[0]],ADP))):
            W = mse(ADP_database[KNN[0]],ADP)
            KNN_Loc = Loc_database[KNN[0]]
            KNN_ADP = ADP_database[KNN[0]]

    for idx in KNN[1:len(KNN)]:
        if not(np.isnan(mse(ADP_database[idx],ADP))):
            W = np.append(W,mse(ADP_database[idx],ADP))
            KNN_Loc = np.append(KNN_Loc, Loc_database[idx], axis = 0)
            KNN_ADP = np.append(KNN_ADP, ADP_database[idx], axis = 0)

    PredADP = np.reshape(Pred_ADP,(1,64,64))
    if not(np.isnan(mse(PredADP,ADP))):
        W = np.append(W,mse(PredADP,ADP))
        KNN_Loc = np.append(KNN_Loc,np.reshape(Pred_Loc,(1,3)), axis = 0)
        KNN_ADP = np.append(KNN_ADP,PredADP, axis = 0)
    Est_Loc = np.zeros((1,3))
    Est_ADP = np.zeros((64,64))
    
    if not(len(W) == 0):
        W = W/(np.sum(W))
        for i in range(0,len(W)):
            Est_Loc = Est_Loc + W[i] * KNN_Loc[i]
            Est_ADP = Est_ADP + W[i] * KNN_ADP[i]
    else:
        Est_Loc = Pred_Loc
        Est_ADP = np.reshape(Pred_ADP,(64,64))
    return Est_Loc, Est_ADP


def Loc_ADP_Recovery_Only_Pred(Pred_ADP):
    Est_Loc = FP_Localizer(Pred_ADP)
    Est_ADP = np.reshape(Pred_ADP,(64,64))
    return Est_Loc, Est_ADP


def FP_Localizer_TF2D(ADP,model):  
    ADP = np.reshape(ADP,(1,64,64,1))
    Prediction = model.predict(ADP)
    return Prediction



def Loc_ADP_Recovery_TF2D(ADP,Pred_ADP,Previous_Loc,Loc_database,ADP_database):
    #plt.figure()
    #plt.imshow(np.reshape(ADP,(64,64)))
    #plt.show()
    KNN = np.where((np.linalg.norm(Loc_database - Previous_Loc, axis = 1 ) < 2))
    #print('len knn = ' + str(len(KNN)))
    KNN = np.asarray(KNN).T
    W = []
    if (np.linalg.norm(FP_Localizer_TF2D(Pred_ADP) - Previous_Loc) > 1):
        Pred_Loc = Previous_Loc
    else:
        Pred_Loc = FP_Localizer_TF2D(Pred_ADP)

    if (len(KNN) > 0) and not(np.isnan(mse(ADP_database[KNN[0]],ADP))):
            W = mse(ADP_database[KNN[0]],ADP)
            KNN_Loc = Loc_database[KNN[0]]
            KNN_ADP = ADP_database[KNN[0]]

    if (len(KNN) > 0):
        for idx in KNN[1:len(KNN)]:
            if not(np.isnan(mse(ADP_database[idx],ADP))):
                W = np.append(W,mse(ADP_database[idx],ADP))
                KNN_Loc = np.append(KNN_Loc, Loc_database[idx], axis = 0)
                KNN_ADP = np.append(KNN_ADP, ADP_database[idx], axis = 0)

    PredADP = np.reshape(Pred_ADP,(1,64,64))
    if (len(KNN) > 0) and not(np.isnan(mse(PredADP,ADP))):
        W = np.append(W,mse(PredADP,ADP))
        KNN_Loc = np.append(KNN_Loc,np.reshape(Pred_Loc,(1,3)), axis = 0)
        KNN_ADP = np.append(KNN_ADP,PredADP, axis = 0)
    Est_Loc = np.zeros((1,3))
    Est_ADP = np.zeros((64,64))
    W1 = np.power(W,100)
    Warg =  np.argsort(W)
    #print(W)
    #print(np.sum(W))
    if not(len(W) == 0):
        #print('I am here')
        WKNNMax = np.where(W == np.max(W))
        #W = W/(np.sum(W))
        if np.sum(W) != 0:     
                W = W/np.sum(W)
        #print(W)
        if np.sum(W) != 0:
            for i in range(0,len(W)):
        #for i in range(0,10):
                Est_Loc = Est_Loc + W[i] * KNN_Loc[i]
                Est_ADP = Est_ADP + W[i] * KNN_ADP[i]
        if np.sum(W) == 0:
            #print('I am ridi')
            Est_Loc = Pred_Loc
            Est_ADP = np.reshape(PredADP[0],(64,64))
    else:
        print('jiiighiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        Est_Loc = Pred_Loc
        Est_ADP = np.reshape(PredADP[0],(64,64))
    #Est_ADP = np.reshape(PredADP[0],(64,64))
    return Est_Loc, Est_ADP



def Loc_ADP_Recovery_Only_Pred_TF2D(Pred_ADP,model1):
    Est_Loc = FP_Localizer_TF2D(Pred_ADP)
    #Est_Loc = CNN_KNN(Pred_ADP,20,50,model1)
    Est_ADP = np.reshape(Pred_ADP,(64,64))
    return Est_Loc, Est_ADP


def mse(imageA, imageB):
    A = np.matrix.flatten(imageA)
    B = np.matrix.flatten(imageB)
  # the 'Mean Squared Error' between the two images is the
  # sum of the squared difference between the two images;
  # NOTE: the two images must have the same dimension
    if (np.linalg.norm(A) == 0) or (np.linalg.norm(B) == 0): 
        err = 0
    else:
        err =(np.sum(np.multiply(A,B)))/(np.linalg.norm(A)*np.linalg.norm(B))
    # return the MSE, the lower the error, the more "similar"
  # the two images are
    return err