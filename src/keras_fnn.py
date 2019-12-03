#!/usr/bin/env python3

'''
ref: https://www.geeksforgeeks.org/prediction-of-wine-type-using-deep-learning/
'''

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

# Read in white wine data 
# white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';') 
white = pd.read_csv('../data/winequality-white.csv', delimiter=';', dtype=float)

# Read in red wine data 
# red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';') 
red = pd.read_csv('../data/winequality-red.csv', delimiter=';', dtype=float)

# split data
# Add `type` column to `red` with price one 
red['type'] = 1

# Add `type` column to `white` with price zero 
white['type'] = 0

# Append `white` to `red` 
wines = red.append(white, ignore_index = True) 

# Import `train_test_split` from `sklearn.model_selection` 
from sklearn.model_selection import train_test_split 
X = wines.ix[:, 0:11] 
y = np.ravel(wines.type) 

# Splitting the data set for training and validating 
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.34, random_state = 45) 

# structure of network
# Import `Sequential` from `keras.models` 
from keras.models import Sequential 

# Import `Dense` from `keras.layers` 
from keras.layers import Dense 

# Initialize the constructor 
model = Sequential() 

# Add an input layer 
model.add(Dense(12, activation ='relu', input_shape =(11, ))) 

# Add one hidden layer 
model.add(Dense(9, activation ='relu')) 

# Add an output layer 
model.add(Dense(1, activation ='sigmoid')) 

# Model output shape 
model.output_shape 

# Model summary 
model.summary() 

# Model config 
model.get_config() 

# List all weight tensors 
model.get_weights() 
model.compile(loss ='binary_crossentropy', 
optimizer ='adam', metrics =['accuracy']) 

# training and prediction
# Training Model 
model.fit(X_train, y_train, epochs = 3, 
        batch_size = 1, verbose = 1) 

# Predicting the Value 
y_pred = model.predict(X_test) 
print(y_pred) 
