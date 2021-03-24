# Main Database Load

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import numpy as np
import warnings
from numpy import load
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import time




import os
import shutil
import argparse
import numpy as np
import torch
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
import sys
from numpy import load
import time
import sys
import csv
import numpy as np



## Check if ADP is distorted or Accurate, Algorithm 1 of the paper
def Loc_Credible(ADP,Loc,Loc_database,ADP_database,Previous_Loc):
    if len(Previous_Loc) > 0 and  np.linalg.norm(Loc - Previous_Loc) > 0.2 :
        out = 'Location is not Credible'
        return out
    KNN = np.where((np.linalg.norm(Loc_database - Loc, axis = 1 ) < 0.05))
    KNN = np.asarray(KNN).T
    ADP = np.reshape(ADP,(1,32,32))
    if len(KNN) == 0:
        out = 'Location is not Credible'
        return out
    for idx in KNN:
        if mse(ADP_database[idx],ADP) > 0.95 and not(np.isnan(mse(ADP_database[idx],ADP))) :
            out = 'Location is Credible'
            return out
    out = 'Location is not Credible'
    return out

## This Function Is the Second Algorithm pf the paper
def Loc_ADP_Recovery_TF2D(ADP,Pred_ADP,Previous_Loc,Loc_database,ADP_database,model):
    #plt.figure()
    #plt.imshow(np.reshape(ADP,(64,64)))
    #plt.show()
    KNN = np.where((np.linalg.norm(Loc_database - Previous_Loc, axis = 1 ) < 0.1))
    #print('len knn = ' + str(len(KNN)))
    KNN = np.asarray(KNN).T
    W = []
    if (np.linalg.norm(FP_Localizer_TF2D(Pred_ADP,model) - Previous_Loc) > 0.1):
        Pred_Loc = Previous_Loc
    else:
        Pred_Loc = FP_Localizer_TF2D(Pred_ADP,model)

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

    PredADP = np.reshape(Pred_ADP,(1,32,32))
    if (len(KNN) > 0) and not(np.isnan(mse(PredADP,ADP))):
        W = np.append(W,mse(PredADP,ADP))
        KNN_Loc = np.append(KNN_Loc,np.reshape(Pred_Loc,(1,3)), axis = 0)
        KNN_ADP = np.append(KNN_ADP,PredADP, axis = 0)
    Est_Loc = np.zeros((1,3))
    Est_ADP = np.zeros((32,32))
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
            Est_ADP = np.reshape(PredADP[0],(32,32))
    else:
        print('jiiighiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        Est_Loc = Pred_Loc
        Est_ADP = np.reshape(PredADP[0],(32,32))
    #Est_ADP = np.reshape(PredADP[0],(64,64))
    return Est_Loc, Est_ADP

## This is the DCNN output
def Loc_ADP_Recovery_Only_Pred_TF2D(Pred_ADP,model):
    Est_Loc = FP_Localizer_TF2D(Pred_ADP,model)
    #Est_Loc = CNN_KNN(Pred_ADP,20,50,model1)
    Est_ADP = np.reshape(Pred_ADP,(32,32))
    return Est_Loc, Est_ADP

## This function is the Normalized Correlation of (7)
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
## Localization using DCNN
def FP_Localizer_TF2D(ADP,model):  
    ADP = np.reshape(ADP,(1,32,32,1))
    Prediction = model.predict(ADP)
    return Prediction

## DCNN Model load
DCNNmodel = tf.keras.models.load_model('DCNNweights/Weights-145--0.01163.hdf5')


## Data Entry
# Output of PredRNN


# Reading Database
training_database_path ='Data/TrainDCNNI1.npz'
database = load(training_database_path)
print(database['ADP'].shape)

# Database data preparation
Loc_database = database['Loc']
ADP_database = database['ADP']
ADP_database = np.reshape(ADP_database,(66550,32,32))

# Reading Test frames
data_path ='Data/testframesI2.npz'
data = load(data_path)

# Predicted Data Preparation
ADP_series = data['ADP']
CLADP_series = data['CLADP']
CLADPL_series = data['CLADPL']
CLADPNL_series = data['CLADPNL']
CLADPANL_series = data['CLADPANL']
Loc_series = data['Loc']
Loc_series = np.reshape(Loc_series,(1000,20,3))



# PredRNN Initialization

__author__ = 'yunbo'


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# training/test
parser.add_argument('--is_training', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')


parser.add_argument('--train_data_paths', type=str, default='')
parser.add_argument('--valid_data_paths', type=str, default='')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=32)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='PredRNN Models/model.ckpt-100000')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=15000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.0002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iterations', type=int, default=15000)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=2500)
parser.add_argument('--snapshot_interval', type=int, default=500)
parser.add_argument('--num_save_samples', type=int, default=2000)
parser.add_argument('--n_gpu', type=int, default=1)

args, unknown = parser.parse_known_args()
print(args)

def mytest(input, model):
    input = np.reshape(input,(1, 10, 32, 32, 1))
    input = np.concatenate((input,np.zeros((1, 10, 32, 32, 1))), axis = 1)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []

    real_input_flag = np.zeros((1, 9, 8, 8, 16))
    out = np.zeros((1,10,32,32,1))
    test_ims = input[0:1]


    test_dat = preprocess.reshape_patch(test_ims, 4)
    img_gen = model.test(test_dat, real_input_flag)

    img_gen = preprocess.reshape_patch_back(img_gen, 4)
    output_length = 10
    img_gen_length = img_gen.shape[1]
    out = img_gen[:, -output_length:]
    return out


print('Initializing models')

## PredRNN Model Init
model = Model(args)

## Data Preparation for LOS Bloackage Scenario
Credible_Est_Loc = np.zeros((1000,20,3))
Credible_Est_ADP = np.zeros((1000,20,32,32))
Loc_Est_Error_CLADPL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_CLADPL = np.zeros((1000,20,1))
Loc_Est_Error_WO_Pred_CLADPL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_PredRNN_CLADPL = np.zeros((1000,10,1))

## PredRNN Model Load
model.load(args.pretrained_model)

## Testing LOS blockage Scenario 
for i in range(0,1000):
        print(i)
        # Reading test data
        test_series1 = np.reshape(CLADPL_series[i,:,:,:],(20,32,32))
        test_series = test_series1
        test_series_Loc = Loc_series[i,:,:]
        Estimated_Loc_FP = np.zeros((20,3))
        PredADP = np.zeros((1, 10, 32, 32, 1))
        start_time = time.time()
        ## Appying DyLoc frame by frame
        for j in range(0,20):
            Estimated_Loc_FP[j] = FP_Localizer_TF2D(test_series1[j:j+1],DCNNmodel)
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
                if j > 9:
                    print('severe warning')
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is not Credible':
                Credible_Est_ADP[i,j] = np.reshape(test_series1[j:j+1],(32,32))
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                print('0 warning')
                if j < 10:
                    print('warning')
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is not Credible':    
                if j < 10:
                    print('warning')                   
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(Credible_Est_ADP[i,j-1],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
                else:
                    PredADP = mytest(Credible_Est_ADP[i,j-10:j], model)
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(PredADP[0, 0, :, :, 0],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
            if j > 9 :
                PredADP = mytest(Credible_Est_ADP[i,j-10:j], model) 
                Est_Loc_temp, Est_ADP_temp = Loc_ADP_Recovery_Only_Pred_TF2D(PredADP[0, 0, :, :, 0],DCNNmodel) 
                Loc_Est_Error_CNN_FP_PredRNN_CLADPL [i,j-10] = np.linalg.norm(Est_Loc_temp - test_series_Loc[j]) 
            Loc_Est_Error_CLADPL[i,j] = np.linalg.norm(Credible_Est_Loc[i,j] - test_series_Loc[j])
            Loc_Est_Error_CNN_FP_CLADPL [i,j] = np.linalg.norm(Estimated_Loc_FP[j] - test_series_Loc[j])
        print("--- %s seconds ---" % (time.time() - start_time))
print('finish')

## Saving Results
np.savetxt('Results/FullSystemerror_LOSBlocked.csv', np.reshape(Loc_Est_Error_CLADPL,(1000,20)), delimiter=',')
np.savetxt('Results/FPerror_LOSBlocked.csv', np.reshape(Loc_Est_Error_CNN_FP_CLADPL,(1000,20)), delimiter=',')
np.savetxt('Results/Prederror_LOSBlocked.csv', np.reshape(Loc_Est_Error_CNN_FP_PredRNN_CLADPL,(1000,10)), delimiter=',')


## Data Preparation for NLOS Bloackage Scenario
Credible_Est_Loc = np.zeros((1000,20,3))
Credible_Est_ADP = np.zeros((1000,20,32,32))
Loc_Est_Error_CLADPNL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_CLADPNL = np.zeros((1000,20,1))
Loc_Est_Error_WO_Pred_CLADPNL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_PredRNN_CLADPNL = np.zeros((1000,10,1))


## Testing NLOS blockage Scenario 
for i in range(0,1000):
        print(i)
        # Reading test data

        test_series1 = np.reshape(CLADPNL_series[i,:,:,:],(20,32,32))
        test_series = test_series1
        test_series_Loc = Loc_series[i,:,:]
        Estimated_Loc_FP = np.zeros((20,3))
        PredADP = np.zeros((1, 10, 32, 32, 1))
        start_time = time.time()

        ## Appying DyLoc frame by frame

        for j in range(0,20):
            Estimated_Loc_FP[j] = FP_Localizer_TF2D(test_series1[j:j+1],DCNNmodel)
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
                if j > 9:
                    print('severe warning')
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is not Credible':
                Credible_Est_ADP[i,j] = np.reshape(test_series1[j:j+1],(32,32))
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                print('0 warning')
                if j < 10:
                    print('warning')
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is not Credible':    
                if j < 10:
                    print('warning')                   
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(Credible_Est_ADP[i,j-1],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
                else:
                    PredADP = mytest(Credible_Est_ADP[i,j-10:j], model)
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(PredADP[0, 0, :, :, 0],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
            if j > 9 :
                PredADP = mytest(Credible_Est_ADP[i,j-10:j], model) 
                Est_Loc_temp, Est_ADP_temp = Loc_ADP_Recovery_Only_Pred_TF2D(PredADP[0, 0, :, :, 0],DCNNmodel) 
                Loc_Est_Error_CNN_FP_PredRNN_CLADPNL [i,j-10] = np.linalg.norm(Est_Loc_temp - test_series_Loc[j]) 
            Loc_Est_Error_CLADPNL[i,j] = np.linalg.norm(Credible_Est_Loc[i,j] - test_series_Loc[j])
            Loc_Est_Error_CNN_FP_CLADPNL [i,j] = np.linalg.norm(Estimated_Loc_FP[j] - test_series_Loc[j])
        print("--- %s seconds ---" % (time.time() - start_time))
print('finish')

## Saving Results
np.savetxt('Results/FullSystemerror_NLOSBlocked.csv', np.reshape(Loc_Est_Error_CLADPNL,(1000,20)), delimiter=',')
np.savetxt('Results/FPerror_NLOSBlocked.csv', np.reshape(Loc_Est_Error_CNN_FP_CLADPNL,(1000,20)), delimiter=',')
np.savetxt('Results/Prederror_NLOSBlocked.csv', np.reshape(Loc_Est_Error_CNN_FP_PredRNN_CLADPNL,(1000,10)), delimiter=',')


## Data Preparation for added NLOS Scenario
Credible_Est_Loc = np.zeros((1000,20,3))
Credible_Est_ADP = np.zeros((1000,20,32,32))
Loc_Est_Error_CLADPANL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_CLADPANL = np.zeros((1000,20,1))
Loc_Est_Error_WO_Pred_CLADPANL = np.zeros((1000,20,1))
Loc_Est_Error_CNN_FP_PredRNN_CLADPANL = np.zeros((1000,10,1))


## Testing Added NLOS Scenario 
for i in range(0,1000):

        print(i)
        # Reading test data

        test_series1 = np.reshape(CLADPANL_series[i,:,:,:],(20,32,32))
        test_series = test_series1
        test_series_Loc = Loc_series[i,:,:]
        Estimated_Loc_FP = np.zeros((20,3))
        PredADP = np.zeros((1, 10, 32, 32, 1))
        start_time = time.time()

        ## Appying DyLoc frame by frame

        for j in range(0,20):
            Estimated_Loc_FP[j] = FP_Localizer_TF2D(test_series1[j:j+1],DCNNmodel)
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
                if j > 9:
                    print('severe warning')
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is Credible':
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                Credible_Est_ADP[i,j] = np.reshape(test_series[j:j+1],(32,32))
            if j == 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,[]) == 'Location is not Credible':
                Credible_Est_ADP[i,j] = np.reshape(test_series1[j:j+1],(32,32))
                Credible_Est_Loc[i,j] = Estimated_Loc_FP[j]
                print('0 warning')
                if j < 10:
                    print('warning')
            if j > 0 and Loc_Credible(test_series1[j:j+1],Estimated_Loc_FP[j],Loc_database,ADP_database ,Credible_Est_Loc[i,j-1]) == 'Location is not Credible':    
                if j < 10:
                    print('warning')                   
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(Credible_Est_ADP[i,j-1],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
                else:
                    PredADP = mytest(Credible_Est_ADP[i,j-10:j], model)
                    Credible_Est_Loc[i,j], CredADP = Loc_ADP_Recovery_TF2D(test_series1[j],np.reshape(PredADP[0, 0, :, :, 0],(1,32*32)),Credible_Est_Loc[i,j-1],Loc_database,ADP_database,DCNNmodel)
                    Credible_Est_ADP[i,j] = np.reshape(CredADP,(32,32))
            if j > 9 :
                PredADP = mytest(Credible_Est_ADP[i,j-10:j], model) 
                Est_Loc_temp, Est_ADP_temp = Loc_ADP_Recovery_Only_Pred_TF2D(PredADP[0, 0, :, :, 0],DCNNmodel) 
                Loc_Est_Error_CNN_FP_PredRNN_CLADPANL [i,j-10] = np.linalg.norm(Est_Loc_temp - test_series_Loc[j]) 
            Loc_Est_Error_CLADPANL[i,j] = np.linalg.norm(Credible_Est_Loc[i,j] - test_series_Loc[j])
            Loc_Est_Error_CNN_FP_CLADPANL [i,j] = np.linalg.norm(Estimated_Loc_FP[j] - test_series_Loc[j])
        print("--- %s seconds ---" % (time.time() - start_time))
print('finish')

## Saving Results

np.savetxt('Results/FullSystemerror_NLOSAdded.csv', np.reshape(Loc_Est_Error_CLADPANL,(1000,20)), delimiter=',')
np.savetxt('Results/FPerror_NLOSAdded.csv', np.reshape(Loc_Est_Error_CNN_FP_CLADPANL,(1000,20)), delimiter=',')
np.savetxt('Results/Prederror_NLOSAdded.csv', np.reshape(Loc_Est_Error_CNN_FP_PredRNN_CLADPANL,(1000,10)), delimiter=',')
