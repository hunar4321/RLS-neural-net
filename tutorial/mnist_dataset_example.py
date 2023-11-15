from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# Load MNIST dataset
mnist = datasets.fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Normalize the pixel values to the range [0, 1]
X /= 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adding a column of ones to the features to include the bias (intercept) in the model
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print('----------------------------')
# Print out the sizes of the training and testing sets
print("Size of training set: {}".format(X_train.shape[0]))
print("Size of testing set: {}".format(X_test.shape[0]))
print('----------------------------')
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
        wx = np.sum(x[i] * x[j]) / (sx[i] + 1e-8)  # Adding a small constant to avoid division by zero
        x[j] -= wx * x[i]

num_classes = len(np.unique(y_train))

### finding the weights of the decorrelated xs with ys
print("Method 1. regression on ys using multiple y_classes in the form of one_hot matrix")
ys = np.zeros((N, num_classes))
for i in range(len(y_train)):  # converting the classes to one_hot format
    ys[i, y_train[i]] = 1
    
wy = np.zeros((M, num_classes))    
for c in range(num_classes):
    y = ys[:, c]
    for i in range(M-1, -1, -1):
        wy[i, c] = np.sum(y * x[i]) / (sx[i] + 1e-8)  # Adding a small constant to avoid division by zero
        y -= wy[i, c] * xs[i]    
    
## predict the training set
yh_train = X_train @ wy
y_train_pred = np.argmax(yh_train, axis=1)
train_accuracy = np.sum(y_train_pred == y_train) / len(y_train)
print('train accuracy:', train_accuracy)

## predict the testing set
yh_test = X_test @ wy
y_test_pred = np.argmax(yh_test, axis=1)
test_accuracy = np.sum(y_test_pred == y_test) / len(y_test)
print('test accuracy:', test_accuracy)

print("---------------------------------")
print("Method 2. regression on ys with simple rounding & thresholding of the predicted y classes.....")

wy = np.zeros(M)
y = y_train.astype(float)    
for i in range(M-1, -1, -1):
    wy[i] = np.sum(y * x[i]) / (sx[i] + 1e-8)  # Adding a small constant to avoid division by zero
    y -= wy[i] * xs[i]

## predict the training set
yh_train = X_train @ wy
y_train_pred = np.round(yh_train).astype(int)
y_train_pred = np.clip(y_train_pred, 0, num_classes-1)
train_accuracy = np.sum(y_train_pred == y_train) / len(y_train)
print('train accuracy:', train_accuracy)

## predict the testing set
yh_test = X_test @ wy
y_test_pred = np.round(yh_test).astype(int)
y_test_pred = np.clip(y_test_pred, 0, num_classes-1)
test_accuracy = np.sum(y_test_pred == y_test) / len(y_test)
print('test accuracy:', test_accuracy)

print("---------------------------------")
print("Method 3. regression on ys using multiple y_classes in the form of random vectors (embeddings)")

np.random.seed(1)
# Generate a random vector (embedding) for each class
embed_size = 10  # can be the same as the class number, 3, or it can be more
class_vectors = np.random.randn(num_classes, embed_size)
ys = np.zeros((N, embed_size))
# Assign each label its corresponding random vector
for i in range(len(y_train)):
    ys[i] = class_vectors[y_train[i]]
    
wy = np.zeros((M, embed_size))    
for c in range(embed_size):
    y = ys[:, c]
    for i in range(M-1, -1, -1):
        wy[i, c] = np.sum(y * x[i]) / (sx[i] + 1e-8)  # Adding a small constant to avoid division by zero
        y -= wy[i, c] * xs[i]    
    
## In method 3, the predicted output will be a vector instead of a single number, so we use a simple distance measure below to find the nearest class vector to the output 
def find_nearest_vector(vec, class_vectors):
    min_distance = np.inf
    min_index = 0
    for i in range(len(class_vectors)):
        distance = np.sum(np.abs(vec - class_vectors[i])) 
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index    
    
## predict the training set
yh_train = X_train @ wy
y_train_pred = [find_nearest_vector(vec, class_vectors) for vec in yh_train]
train_accuracy = np.sum(y_train_pred == y_train) / len(y_train)
print('train accuracy:', train_accuracy)

## predict the testing set
yh_test = X_test @ wy
y_test_pred = [find_nearest_vector(vec, class_vectors) for vec in yh_test]
test_accuracy = np.sum(y_test_pred == y_test) / len(y_test)
print('test accuracy:', test_accuracy)
