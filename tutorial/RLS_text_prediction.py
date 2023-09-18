## a simple example usage for a multiclass example like predicting the next letter

import numpy as np
import matplotlib.pylab as plt

text = '''
01 02 03 04 05 06 07 08 09 10
11 12 13 14 15 16 17 18 19 20
21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48 49 50
51 52 53 54 55 56 57 58 59 60
61 62 63 64 65 66 67 68 69 70
71 72 73 74 75 76 77 78 79 80
81 82 83 84 85 86 87 88 89 90
'''    
     
N = len(text) 
M = 10
emb_size = 100

words=text[:N]
chars = sorted(list(set(words)))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)  
xs = np.zeros(N, dtype=int)
for i in range(N):
    xs[i] = stoi[words[i]]
     
xemb = np.random.randn(vocab_size, emb_size) #embedding the letters
xpos = np.random.randn(M, emb_size) #positional embedding
yind = np.eye(vocab_size) #classes

wx = np.zeros((emb_size, emb_size))
wy = np.zeros((vocab_size, emb_size))
xy = np.zeros((vocab_size, emb_size))
xx = np.zeros((emb_size, emb_size))
sx = np.zeros(emb_size) + 0.000001
yh = np.zeros(N)
ys = np.zeros(N, dtype = int)
for n in range(N-(M+1)):
    
    if n%100==0:
        print(".", end="")
      
    # prepare the inputs
    x = xs[n:n+M].copy()
    x = xemb[x] * xpos
    x = np.sum(x, axis=0)
    
    # decorrelating the xs
    for i in range(emb_size):
        sx[i] += (x[i]**2)
        for j in range(i+1, emb_size):
            xx[i, j] += (x[i] * x[j])
            wx[i, j] = xx[i, j] / sx[i]
            x[j] -= wx[i, j] * x[i]
       
    #predict 
    pred = wy @ x
    yh[n] = np.argmax(pred)
    
    # regressing the xs on the ys (outputs)
    ys[n] = xs[n+M]   
    for j in range(vocab_size):
        yi = yind[ys[n], j]
        for i in range(emb_size):   
            xy[j,i] += yi * x[i]
            wy[j,i] = xy[j, i] / sx[i]
            yi -= wy[j, i] * x[i]  
                    
plt.figure(1)        
plt.title("Prediction")
plt.plot(ys)
plt.plot(yh)
plt.legend(["truth", "prediction"])  

sen = ""
for i in range(len(yh)):
    sen += itos[yh[i]]
print(sen)
