from numpy import load
import time
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ADP Dataset for DCNN Training
file_ADP = 'Data/TrainDataADP.csv'
ADP = np.array(pd.read_csv(file_ADP, header = None))
input_raw_data = np.reshape(ADP,(181*1100,1,64,64))
print(ADP.shape)

## Paired Loc Dataset for DCNN Training
file_Loc = 'Data/TrainDataLoc.csv'
Loc = np.array(pd.read_csv(file_Loc, header = None))
Loc = np.reshape(Loc,(181*1100,3))
print(Loc.shape)

## Saving Training Dataset for DCNN
np.savez_compressed('TrainDCNN.npz', ADP = input_raw_data, Loc = Loc)

## ADP test frames accurate
file = 'Data/testframes_CLADP_O13p5.csv'
M1 = np.array(pd.read_csv(file, header = None))
M1 = np.float32(M1)

## ADP test frames Distorted LOS Blockage
file = 'Data/testframes_CLADPL_O13p5.csv'
M2 = np.array(pd.read_csv(file, header = None))
M2 = np.float32(M2)


## ADP test frames Distorted NLOS Blockage
file = 'Data/testframes_CLADPNL_O13p5.csv'
M3 = np.array(pd.read_csv(file, header = None))
M3 = np.float32(M3)

## ADP test frames Distorted NLOS Addition
file = 'Data/testframes_CLAddDPNL_O13p5.csv'
M4 = np.array(pd.read_csv(file, header = None))
M4 = np.float32(M4)

## Paired Locations
file = 'testforpython/testADPAddNL_Loc.csv'
M5 = np.array(pd.read_csv(file, header = None))
M5 = np.float32(M5)

## Saving Testframes for DyLoc
np.savez_compressed('testframes.npz', CLADP = M1, CLADPL = M2, CLADPNL = M3, CLADPANL = M4, Loc = M5)

## Loading the structure of the dataset for PredRNN
data = load('moving-mnist-train.npz')

## Loading ADP frames for PredRNN Training
file = 'Moving_ADP_O13p5.csv'
M = np.array(pd.read_csv(file, header = None))
M = np.reshape(M,(200000,1,64,64))
np.savez_compressed('moving-ADP-train-O1.npz', clips = data['clips'], dims = data['dims'] , input_raw_data = M)

data = load('moving-mnist-test.npz')

## Loading ADP frames for PredRNN Testing
file = 'Moving_ADP_test_O13p5.csv'
M = np.array(pd.read_csv(file, header = None))
M = np.reshape(M,(40000,1,64,64))
np.savez_compressed('moving-ADP-test-O1.npz', clips = data['clips'], dims = data['dims'] , input_raw_data = M)
