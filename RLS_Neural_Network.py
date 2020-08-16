# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 00:51:38 2020

@author: Hunar Ahmad
"""

import numpy as np

class RlsNode:
    """
    Recursive Least Squares Estimation (This is the update part of Kalman filter)
    """
    def __init__(self, _m, _name):
        self.name = _name;
        self.M = _m;
        self.w = np.random.rand(1, _m);
        self.P = np.eye(_m)/1;

    def RlsPredict(self, v):
        return np.dot(self.w , v);
    
    def RlsLearn(self, v, e):
        pv = np.dot(self.P, v)
        vv = np.dot(pv.T, v)        
        eMod = pv/vv;
        ee = eMod * e;        
        self.w = self.w + ee
        outer = np.outer(eMod , pv);
        self.P = self.P - outer

        
class Net:
    """ 
    neural network (single hidden layer where the weights in first layer (iWh) are randomly initialized)
    """
    def __init__(self, _input_size, _neurons):
        self.input_size = _input_size;
        self.neurons = _neurons;
        self.iWh = (np.random.rand(_neurons, _input_size)-0.5)*1
        self.nodes = [];
        
    def CreateOutputNode(self, _name):
        nn = RlsNode(self.neurons, _name);
        self.nodes.append(nn)
        
    def sigmoid(self, x):
        return 1 / (1 + np.e ** -x)

    def FeedForwardL1(self, v):
        vout = np.dot(self.iWh, v);
        # tout = np.tanh(vout);
        tout = self.sigmoid(vout);
        return tout + 0.00000001 ## adding a small value to avoid dividion by zero in the upcoming computations!
    
    ### RLS layer (Trainable weights using RLS algorthim)
    def FeedForwardL2(self, tout):
        yhats = [];
        for i in range(len(self.nodes)):
            p = self.nodes[i].RlsPredict(tout);
            yhats.append(p[0]);
        return np.asarray(yhats)
    
    ### Error Evaluation
    def Evaluate(self, ys, yhats):
        errs = ys - yhats
        return errs
    
    def Learn(self, acts, errs):
        for i in range(len(self.nodes)):
            self.nodes[i].RlsLearn(acts, errs[i]) #change to errs[0][i] if indexing error happen


# #### Example Usage ###            

x = [[1, 1],[ 0, 0],[ 1, 0],[ 0,1]]; ## input data
y = [1, 1, 0, 0]; ## output data (targets)

## configuring the network
n_input = 2
n_neurons = 5
n_output = 1
net = Net(n_input, n_neurons)
for i in range(n_output):
    net.CreateOutputNode(i)

## training
N = len(x) ## you only need one iteration over the samples to learn (RLS is a near one-shot learning!)
for i in range(N):
    inputt = x[i][:]
    L1 = net.FeedForwardL1(inputt);
    yhats = net.FeedForwardL2(L1)
    errs = net.Evaluate(y[i], yhats)
    net.Learn(L1, errs)

## evaluate after learning
yh= [];
for i in range(N):
    inputt = x[i][:] 
    L1 = net.FeedForwardL1(inputt);
    yhats = net.FeedForwardL2(L1)
    
    print("input", inputt, "predicted output:", yhats[0])
  
   


    