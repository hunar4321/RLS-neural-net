"""
Lattice Recursive Least Squares for predicting next elements in the squential data 
Video Tutorial: https://youtu.be/4vGaN1dTVhw
"""

import numpy as np
import matplotlib.pylab as plt

# generating time series data
N = 200
t = np.arange(0.0001, 1.001, 0.001)
data = 2 * np.sin(2 * np.pi * 50 * t)  
ys = data[:N]
for i in range(1, N, 2):
    ys[i] = 1
    
#ys[70] = -1
#ys[150] = -1

M = 20 # looking back M steps to predict the signal
b = np.zeros(M)
t = np.zeros(M)
f = np.zeros(M)
wf = np.zeros(M)
wb = np.zeros(M)
fb = np.zeros(M) 
ff = np.zeros(M) + 0.000001
bb = np.zeros(M) + 0.000001
e = np.zeros(N)

pred = np.zeros(N)
act = np.zeros((M, N))
for n in range(N - 1):

    pred[n] = wf @ b        
        
    f[0] = ys[n]
    t[0] = ys[n]  

    for m in range(M-1):
        
        ff[m] += f[m] * f[m]
        bb[m] += b[m] * b[m]
        fb[m] += f[m] * b[m]

        wb[m] = fb[m] / ff[m]
        wf[m] = fb[m] / bb[m]

        f[m + 1] = f[m] - wf[m] * b[m]
        t[m + 1] = b[m] - wb[m] * f[m]
          
    b = t.copy()
        
    e[n] = f[-1] 
    act[:, n] = f

plt.figure(1)
plt.plot(ys)
plt.plot(pred)
plt.legend(["truth", "prediction"])  

plt.figure(2)
plt.title("Error")
plt.plot(e)

plt.figure(4)
plt.title("Activity")
plt.imshow(np.flipud(np.abs(act)))

###### generating data ############

# generated_data = np.zeros(N)
# for n in range(N - 2):

#     generated_data[n] = wf @ b        
        
#     f[0] = generated_data[n]
#     t[0] = generated_data[n]  

#     for m in range(M-1):
        
#         f[m + 1] = f[m] - wf[m] * b[m]
#         t[m + 1] = b[m] - wb[m] * f[m]
          
#     b = t.copy()
        
# plt.figure(3)
# plt.title("generated_data")
# plt.plot(generated_data)

