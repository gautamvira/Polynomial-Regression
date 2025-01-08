import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import cProfile
import pstats

def getData(N, var):                          #function to generate datasets
  df = pd.DataFrame(columns=list('XY'))
  x = []
  y = []
  for i in range(N):                          #for N sample sizes
    x.append(np.random.random())              #get random x
    y.append(math.cos(math.pi * 2 * x[i]) + np.random.normal(0, var))    #get Y 
  x = np.array(x)
  y = np.array(y)
  df = pd.DataFrame({'X':x, 'Y':y})
  return df

def getMSE(df, Y):                            #Function to get MSE from y_hat and y
  ytrue = df['Y'].to_numpy()
  MSE = np.square(ytrue - Y).mean()
  return MSE

def getPoly(x, d):                            #function to get x-polynomials of degree y
  xpoly = []
  for j in range(len(x)):
    xi= []
    for c in range(d+1):                      #for degrees 0 to d
      xi.append(x[j] **c)
    xpoly.append(xi)
  return np.array(xpoly)

def fitGD(N, d, df):                          #function to estimate coefficients using gradient descent
  y = df['Y'].to_numpy()
  x = df['X'].to_numpy()
  lr = 0.01
  coeff = np.ones(d+1)
  xpoly = getPoly(x, d)
  wd = 0.1
  coeff = torch.tensor(coeff, requires_grad = True, dtype = torch.float64)        #initializing torch tensors for differentiation calculation
  xpoly = torch.tensor(xpoly, dtype = torch.float64)
  diff = 1
  count = 0
  while np.linalg.norm(diff) > 0.01 and count < 1000:                             #while gradient is not low enough and epochs aren't over  
    loss = ((torch.tensor(y) - (torch.matmul(xpoly, coeff)))**2).mean() + wd * (torch.sum(coeff ** 2)) #loss with weight decay regularization
    loss.backward()
    diff = coeff.grad                                                             #calculate partial gradients
    diff *= lr
    coeff = coeff.detach() - diff                                                 #get coefficients
    coeff.requires_grad=True
    count+=1
  ypred = torch.matmul(xpoly, coeff)                                              #get y_hat values
  return ypred.detach().numpy(), coeff.detach().numpy()

def fitData(N, d, var):                                                           #runner function to fit polynomials and get MSE               
  tdf = getData(N, var)
  yhat, coeff = (fitGD(N, d, tdf))
  MSEin = getMSE(tdf, yhat)
  ttdf = getData(1000, var)                                                       #testing
  x = ttdf['X'].to_numpy()
  xpoly = getPoly(x, d)
  ypred = np.matmul(xpoly, coeff)
  MSEout = getMSE(ttdf, ypred)
  return coeff, MSEin, MSEout

def experiment(d, N, var):                                                        #experimenting function for different combinations
  eout = 0
  ein = 0
  coeffs = []
  for i in range(50):
    coeff, MSEin, MSEout = fitData(N, d, var)
    ein += MSEin
    coeffs.append(coeff)
    eout += MSEout
  mcoeff = (sum(coeffs))/len(coeffs)
  ttdf = getData(1000, var)                                                       #getting Ebias
  x = ttdf['X'].to_numpy()
  xpoly = getPoly(x, d)
  ypred = np.matmul(xpoly, mcoeff)
  ebias = getMSE(ttdf, ypred)
  avgin = ein/30
  avgout = eout/30
  return ebias, avgin, avgout


def runexp():                                                                 #runner function for experiment()
  N = [2, 5, 10, 20, 50, 100, 200]
  d = list(range(0, 20+1))
  var = [0.01, 0.1, 1]
  eb, ein, eo = 0.000000000000, 0.000000000000, 0.000000000000
  for j in N:
    evarin, evarout, evarbias = [], [], []
    for k in var:
      ein, eout, ebias = [], [], []
      for i in d:
        eb, ei, eo = experiment(i, j, k)
        ein.append(ei)
        ebias.append(eb)
        eout.append(eo)
      evarin.append(ein)
      evarout.append(eout)
      evarbias.append(ebias)


    for idx in range(len(evarin)):                                          #plotting graphs for d as x-axis
      plt.figure()
      x1 = list(d)
      y1 = evarin[idx]
      # plotting the line 1 points 
      plt.plot(x1, y1, label = "Ein")
    
      # line 2 points
      x2 = list(d)
      y2 = evarout[idx]
  # plotting the line 2 points 
      plt.plot(x2, y2, label = "Eout")

      x3 = list(d)
      y3 = evarbias[idx]
      plt.plot(x3, y3, label = "Ebias")
  # naming the x axis
      plt.xlabel('d')
  # naming the y axis
      plt.ylabel('E')
  # giving a title to my graph
      plt.title("N = %d and var = %f" %(j, var[idx]))

      plt.legend()
  # function to show the plot
  
  plt.show()

runexp()

"""
#runexp()                                                                
# #running code                                                                         #profiling program for time complexity
with cProfile.Profile() as pr:
  runexp()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
"""

