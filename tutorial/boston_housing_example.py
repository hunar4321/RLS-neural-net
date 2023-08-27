import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import read_csv

# download housing dataset from: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv("housing.xls", header=None, delimiter=r"\s+", names=column_names)

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
xs = data.loc[:,column_sels]
ys = data['MEDV']

xs = xs.to_numpy().T
ys = ys.to_numpy()

## add a bias term
xs = np.vstack( (xs , np.ones(xs.shape[1] ) ) )

N = xs.shape[1]
M = xs.shape[0]

x = xs.copy()  #make copies of xs & ys before modifying them 
y = ys.copy()
wy = np.zeros(M)
sx = np.zeros(M)
for i in range(M):
    sx[i] = np.sum(x[i]**2)
    for j in range(i+1, M):
        wx = np.sum(x[i] * x[j]) /sx[i]
        x[j] -= wx * x[i]
        
for i in range(M-1,-1, -1):
    wy[i] = np.sum(y * x[i]) / sx[i]
    y -= wy[i] * xs[i]
    
yh = wy @ xs #prediction
plt.figure(1)
plt.title("Prediction")
plt.plot(ys)
plt.plot(yh)
plt.legend(["truth", "prediction"])   

print("Solutions:", wy)
 
### comparing the method with matrix inversion using svd & pinv
# w_pinv = np.linalg.pinv(xs.T) @ ys
# plt.figure(2)
# plt.plot(w_pinv)
# plt.plot(wy)



