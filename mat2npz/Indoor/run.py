from numpy import load
import time
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Path  C:\Users\Nazanin\OneDrive - University of Central Florida\predrnn-pytorch-master\data\moving-mnist-example\dataforpython\test_valid_half_LOSBADP.csv
file_ADP = 'Data/TrainDataADP.csv'
ADP = np.array(pd.read_csv(file_ADP, header = None))
input_raw_data = np.reshape(ADP,(121*550,1,32,32))
print(ADP.shape)


file_Loc = 'dataforpython/test_valid_half_NLOSAADP_Loc.csv'
Loc = np.array(pd.read_csv(file_Loc, header = None))
Loc = np.reshape(Loc,(121*550,3))
print(Loc.shape)

np.savez_compressed('TrainDCNN.npz', ADP = input_raw_data, Loc = Loc)


file = 'Data/testframes_CLADP_O13p5.csv'
M1 = np.array(pd.read_csv(file, header = None))
M1 = np.float32(M1)


file = 'Data/testframes_CLADPL_O13p5.csv'
M2 = np.array(pd.read_csv(file, header = None))
M2 = np.float32(M2)

file = 'Data/testframes_CLADPNL_O13p5.csv'
M3 = np.array(pd.read_csv(file, header = None))
M3 = np.float32(M3)


file = 'Data/testframes_CLADPNL_O13p5.csv'
M4 = np.array(pd.read_csv(file, header = None))
M4 = np.float32(M4)


file = 'testforpython/testADPAddNL_Loc.csv'
M5 = np.array(pd.read_csv(file, header = None))
M5 = np.float32(M5)


np.savez_compressed('testframes.npz', CLADP = M1, CLADPL = M2, CLADPNL = M3, CLADPANL = M4, Loc = M5)