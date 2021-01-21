# Main Database Load

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import warnings
from numpy import load
import time
import sys
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import math


#grid size
xnum = 18
ynum = 55

# MSE is use similarity measure between two images
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
 
    return err

# CNN_KNN is a 2 part classifier. First it uses trained CNN network to classify the ADP to a grid. 
# Then it uses WKNN to do a search within the grid to further improve accuracy. 
#Input: ADP, size of grid and trained CNN Model (classifier for first part)
#Output: estiamted (x,y) location corresponding to the input ADP. 
def CNN_KNN(ADP,xnum,ynum,model1):
    
  num_classes = xnum * ynum #number of class
  tot_accuracy = np.zeros(num_classes)
  k = 3 # k in knn 
  csort = (-1)*np.ones(k)
  xtest = ADP
  
  # Model predicts with grid (class) the ADP belongs to. 
  # This is the first part of the classifier (CNN)
  xtest = np.reshape(xtest,(1,32,32,1))
  ypred = model1.predict(xtest)
  
  pred = np.argmax(ypred, axis = 1) 
  
  #Load training data computed in "MakeADPgrid.py"
  # Only loads data for the given class computed in ypred. 
  # For example: If ypred = 3, it will load all ADPs that are contianed within class 3 and their (x,y) location
  train_ADP = np.load('ADP/ADP'+str(int(pred))+'.npy')
  train_c = np.load('ADP/class'+str(int(pred))+'.npy')
  xtrain = np.load('ADP/xtrain'+str(int(pred))+'.npy')
  ytrain = np.load('ADP/ytrain'+str(int(pred))+'.npy')
    
   # initializng accuracy and number of correctly classified samples to zero
  accuracy = 0
  num_correct = 0
  similarity = np.zeros(len(train_ADP))
   # compute similarity between the ADP input sample (to be classified) and all the samples in the ypred class
  for tr in range(len(train_ADP)):
    x = xtest
    y = train_ADP[tr,:,:]
    #similarity[tr] = jadsc(x, y)
    similarity[tr] = mse(x, y)
  
  # Find k (k=3) number of ADPs with the largest similarity to the input (test) ADP
  sim_sort = np.sort(similarity)
  sim_kmax = sim_sort[-k:]
    
   # get (x,y) location corresponding to the k ADPs found above
  csort_x = np.zeros((k))
  csort_y = np.zeros((k))
  ADPout = [k,32,32]
  for i in range(k):
    for j in range(len(similarity)):
      if (similarity[j]==sim_kmax[i]):
        csort_x[i] = xtrain[j]
        ADPout[i] = train_ADP[j]
        #print(xtrain[j])
        csort_y[i] = ytrain[j]
        #print(ytrain[j])
    
  xest = 0
  yest = 0
  # Using similarity as weights, estimate location using weighted-k-nearest-nighbot (WKNN) 
  # This is the second part of the classifier (WKNN)
  w = np.zeros(k)
  if (sum(sim_kmax)!= 0):
    for i in range(k):
        w[i] = sim_kmax[i]/sum(sim_kmax)
    for i in range(k):
        xest = xest + w[i]*csort_x[i]
        yest = yest + w[i]*csort_y[i]
  else:
    xest = xtrain[0]
    xest = ytrain[0]
  # function returns estimates (x,y) location. 
  return [xest, yest,2]


# Test frames Load
data_path ='Data/testframesI2.npz'
data = load(data_path)

# Predicted Data Preparation
ADP_series = data['ADP']
CLADP_series = data['CLADP'] # Samples with no blockage
CLADPL_series = data['CLADPL'] #Samples with LOS blocakge
CLADPNL_series = data['CLADPNL'] #Samples with NLOS blockage
CLADPANL_series = data['CLADPANL'] #Samples with NLOS addition
Loc_series = data['Loc']
Loc_series = np.reshape(Loc_series,(1000,20,3))

model = tf.keras.models.load_model('KNNCNNW1/Weights-296--0.93764.hdf5')

Loc_Est_Error_CLADPL = np.zeros((1000,20,1))
Loc_Est_Error_CLADPNL = np.zeros((1000,20,1))
Loc_Est_Error_CLADPANL = np.zeros((1000,20,1))

Estimated_Loc_KNN_CNN1 = np.zeros((20,3))
Estimated_Loc_KNN_CNN2 = np.zeros((20,3))
Estimated_Loc_KNN_CNN3 = np.zeros((20,3))

#For all samples in the test data, estimate the location corresponing to the ADP input and calculae the error
for i in range(0,1000):
    print(i)
    test_series1 = np.reshape(CLADPL_series[i,:,:,:],(20,32,32)) 
    test_series2 = np.reshape(CLADPNL_series[i,:,:,:],(20,32,32))
    test_series3 = np.reshape(CLADPANL_series[i,:,:,:],(20,32,32))
    test_series_Loc = Loc_series[i,:,:]
    for j in range(0,20):
        Estimated_Loc_KNN_CNN1[j] = CNN_KNN(test_series1[j:j+1],xnum,ynum,model)
        Estimated_Loc_KNN_CNN2[j] = CNN_KNN(test_series2[j:j+1],xnum,ynum,model)
        Estimated_Loc_KNN_CNN3[j] = CNN_KNN(test_series3[j:j+1],xnum,ynum,model)
        Loc_Est_Error_CLADPL[i,j] = np.linalg.norm(Estimated_Loc_KNN_CNN1[j] - test_series_Loc[j])
        Loc_Est_Error_CLADPNL[i,j] = np.linalg.norm(Estimated_Loc_KNN_CNN2[j] - test_series_Loc[j])
        Loc_Est_Error_CLADPANL[i,j] = np.linalg.norm(Estimated_Loc_KNN_CNN3[j] - test_series_Loc[j])

np.savetxt('Results/KNNCNNerror_LOSBlocked.csv', np.reshape(Loc_Est_Error_CLADPL,(1000,20)), delimiter=',')
np.savetxt('Results/KNNCNNerror_NLOSBlocked.csv', np.reshape(Loc_Est_Error_CLADPNL,(1000,20)), delimiter=',')
np.savetxt('Results/KNNCNNerror_NLOSAdded.csv', np.reshape(Loc_Est_Error_CLADPANL,(1000,20)), delimiter=',')
