import numpy as np

xnum = 18
ynum = 55


def get_valid_data(data,xnum,ynum,t):
  num_classes = xnum*ynum
  adp = data['ADP']
  print('adp shape ',adp.shape)
  #get train data
  M = np.reshape(adp,(-1,32,32))
  if t ==0 :
    loc = data['Location']
  elif t ==1:
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

  #This creates a class
  for i in range(len(loc)):
      c[i] = (xnew[i]-1)+xnum*(ynew[i]-1)
  c = np.reshape(c,(len(c),1))

  return loc,c, M



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
  # find all values of subclass
  j=0

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

  for i in range(len(c1)):
      c1[i] = (xnew[i]-1)+xnum*(ynew[i]-1)

  c1 = np.reshape(c1,(n,1))

  return Mnew, c1,xlocnew,ylocnew


data_path1 ='Data/DCNN-train.npz'
data1 = np.load(data_path1)

loc_t, c_t, ADP_t= get_valid_data(data1,xnum,ynum,1)
num_classes = xnum*ynum
x_loc = loc_t[:,0]
y_loc = loc_t[:,1]

for sub in range(num_classes):
   train_ADP, train_c, xtrain, ytrain = get_sub(sub,ADP_t,c_t,x_loc,y_loc)   
   np.save('ADP/ADP'+str(sub),train_ADP)
   np.save('ADP/class'+str(sub),train_c)
   np.save('ADP/xtrain'+str(sub),xtrain)
   np.save('ADP/ytrain'+str(sub),ytrain)