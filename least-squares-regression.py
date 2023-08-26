"""
Multiple Regression Using Recursive Error prediction. 
Author: Hunar Ahmad @ Brainxyz
Video tutorial: https://youtu.be/4vGaN1dTVhw
"""
import numpy as np
import matplotlib.pylab as plt

N = 100 #number of the subjects
M = 10 #number of the variables
xs = np.random.randn(M, N)
ws = np.random.randn(M)
ys = ws @ xs

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
plt.title("Prediction")
plt.plot(ys)
plt.plot(yh)
plt.legend(["truth", "prediction"])   

print("Solutions:", wy)
print("------------------")
print("True weights:", ws)
