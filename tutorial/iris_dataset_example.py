from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adding a column of ones to the features to include the bias (intercept) in the model
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print('De-correlating all the xs with each other')
print('----------------------------')

xs = X_train.T
N = xs.shape[1]
M = xs.shape[0]
x = xs.copy() 
sx = np.zeros(M)
for i in range(M):
    sx[i] = np.sum(x[i]**2)
    for j in range(i+1, M):
        wx = np.sum(x[i] * x[j]) /sx[i]
        x[j] -= wx * x[i]
         
### finding the weights of the decorrelated xs with ys
print("1. regression on ys using multiple y_classes in the form of one_hot matrix")
num_classes = np.max(y_train)+1
ys = np.zeros((N, num_classes))
for i in range(len(y_train)): #converting the classes to one_hot format
    ys[i,  y_train[i]]=1
    
wy = np.zeros((M, num_classes))    
for c in range(num_classes):
    y = ys[:, c]
    for i in range(M-1,-1, -1):
        wy[i, c] = np.sum(y * x[i]) / sx[i]
        y -= wy[i, c] * xs[i]    
    
## predict the training set
yh_train = X_train @ wy
y_train_pred = np.argmax(yh_train, axis=1)
train_accuracy = np.sum(y_train_pred == y_train)/len(y_train)
print('train accuracy:', train_accuracy)

## predict the testing set
yh_test = X_test @ wy
y_test_pred = np.argmax(yh_test, axis=1)
test_accuracy = np.sum(y_test_pred == y_test)/len(y_test)
print('test accuracy:', test_accuracy)


print("---------------------------------")
print("2. regression on ys with simple thresholing of the y calsses.....")

wy = np.zeros(M)
y = np.asarray(y_train, dtype=float)        
for i in range(M-1,-1, -1):
    wy[i] = np.sum(y * x[i]) / sx[i]
    y -= wy[i] * xs[i]

## predict the training set
yh_train = X_train @ wy
y_train_pred = np.round(yh_train).astype(int)
y_train_pred = np.clip(y_train_pred, 0, 2)
train_accuracy = np.sum(y_train_pred == y_train)/len(y_train)
print('train accuracy:', train_accuracy)


## predict the testing set
yh_test = X_test @ wy
y_test_pred = np.round(yh_test).astype(int)
y_test_pred = np.clip(y_test_pred, 0, 2)
test_accuracy = np.sum(y_test_pred == y_test)/len(y_test)
print('test accuracy:', test_accuracy)
