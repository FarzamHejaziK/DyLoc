import numpy as np

# Define x and y grid size. Must match xnum and ynum in the Train.py
xnum = 18
ynum = 55

#Function gets validation data and returns (ADP, class) pairs
# Inputs: raw validation data (ADP, location), and size of (x,y0 grid
# t - handles ame of location dataset. Can be removed if name of locaiton column is known
#Outputs: (ADP, class) pairs and new location based on classes (similar to get_data in tarin.py)
def get_valid_data(data,xnum,ynum,t):
  # calculate total number of classes
  num_classes = xnum*ynum
  adp = data['ADP']
  print('adp shape ',adp.shape)
  # Get DP and (x,y0 location from dataset
  M = np.reshape(adp,(-1,32,32))
  if t ==0 :
    loc = data['Location']
  elif t ==1:
    loc = data['Loc']
  x = loc[:,0]
  y = loc[:,1]
  # Create placeholders for class (grid) and new (X,Y) coordiantes which are at the center of each grid.
  # Example: If the user is in the mth segment in x-direaction and nth segment in y-direction, the new coordinates are (m,n) and the sample belongs to class c = m*n.
  cnew = np.zeros((len(loc),num_classes))
  c = np.zeros(len(loc))
  xnew = np.zeros(len(x))
  ynew = np.zeros(len(y))
  
   # Calcualte step size based on the grid
  xmax = max(x)
  xmin = min(x)
  xstep = (xmax-xmin)/xnum
  ymax = max(y)
  ymin = min(y)
  ystep = (ymax-ymin)/ynum

  # Convert x coordinate to grid segment in x
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

 # Convert y coordinate to grid segment in y
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

   # Assigns classes to dataset based on grid starting with grid(1,1) assigned to c c=1 
   # and grid (m,n) assigned to class c = m*n.
  for i in range(len(loc)):
      c[i] = (xnew[i]-1)+xnum*(ynew[i]-1)
  c = np.reshape(c,(len(c),1))

   # M - ADP
   # c - class
  # location (x,y)
  return loc,c, M


# For each grid, collect all ADP samples from the training set that belong to the same grid into one group. 
# This means that grid will have a collection of ADPs that belong to this grid and this collection will be used for KNN. 
def get_sub(subclass,M,c,x,y):
  xnum = 18
  ynum = 55
  #get train data 
  n = 0
  for i in range(len(c)):
    if (c[i]==subclass):
      n+=1
  M  = np.reshape(M,[-1,64,64])
  Mnew = np.zeros([n,64,64])
  xlocnew = np.zeros(n)
  ylocnew = np.zeros(n)
  
  j=0
  
# for each class i compute the ADP, and (x,Y) location
  for i in range((len(c))):
    if (c[i]==subclass):
      Mnew[j,:,:]=M[i,:,:]
      xlocnew[j] = x[i]
      ylocnew[j] = y[i]
      j+=1

  c1 = np.zeros(n)
  xnew = np.zeros(n)
  ynew = np.zeros(n)
  xmax = max(xlocnew)
  xmin = min(xlocnew)
  xstep = (xmax-xmin)/xnum
  ymax = max(ylocnew)
  ymin = min(ylocnew)
  ystep = (ymax-ymin)/ynum
  
  #computes new x and y locationa based on the grif
  for i in range(n):
      for j in range(1,xnum+1):
          low = xmin + (j-1)*xstep
          high = xmin + j * xstep
          if(xlocnew[i]>=low and x[i]<high):
              xnew[i] = j
          if(xlocnew[i] == xmax):
              xnew[i] =xnum

  for i in range(n):
      for j in range(1,ynum+1):
          low = ymin + (j-1)*ystep
          high = ymin + (j*ystep)
          if(ylocnew[i]>=low and ylocnew[i]<high):
              ynew[i] = j
          if(y[i] == ymax):
              ynew[i] = ynum
  #computes class based on grid
  for i in range(len(c1)):
      c1[i] = (xnew[i]-1)+xnum*(ynew[i]-1)

  c1 = np.reshape(c1,(n,1))
  # Mnew -ADP
  # c1- class
  #xlocnew and ylocnew are x and y coordinates
  return Mnew, c1,xlocnew,ylocnew

#loads Train Dataset
data_path1 ='Data/TrainDCNNI1.npz'
data1 = np.load(data_path1)

#Takes the dataset and grid size as input
#outputs ADP, class, and location(x,y)
loc_t, c_t, ADP_t= get_valid_data(data1,xnum,ynum,1)
num_classes = xnum*ynum
x_loc = loc_t[:,0]
y_loc = loc_t[:,1]

for sub in range(num_classes):
  #For every class on the grid it creates a dataset of ADP, class, and locatioon
   train_ADP, train_c, xtrain, ytrain = get_sub(sub,ADP_t,c_t,x_loc,y_loc)   
   np.save('ADP/ADP'+str(sub),train_ADP)
   np.save('ADP/class'+str(sub),train_c)
   np.save('ADP/xtrain'+str(sub),xtrain)
   np.save('ADP/ytrain'+str(sub),ytrain)
