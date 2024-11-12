import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC, Recall
import pandas as pd
import keras_tuner as kt
import os
from os import listdir
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
from numpy.random import seed
import random
from mlxtend.data import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

#now, all libraries imported, and the variables to train and test are created.
seed(33)
# REQUIREMENTS:
    # fully-connected layer, 56 nodes, ReLU activation
    # fully-connected layer, 10 nodes, softmax activation

#1) Import the image data from my directory. 60,000 imgs, test set 10,000 imgs. All 28 by 28 grayscale.
img_data_directory = '/home/leozj/ecs170/homework-2-Lumiho/fashion-mnist'

#2) Make sure the data is sized correctly and normalized. Otherwise, resize and modify
    # it's already 28x28 grayscale. 10 labels. 
    

#3) Create convolutional layer, pool, another conv layer. create fxn that allows play with the hyperparameters
    # Adding Conv layer: layers.Conv2D([# of filters], (x,x), activation = [fxn], data_format = 'channels_last', input_shape = [dims])
    # Pooling: layers.MaxPooling((x,x))   
    # Create Dense layer (fully-connected layer): layers.Dense(units, activation=[fxn or none])
    # hp.Int() to search for best hyperparameters instead of stating it. define the range. units == number of neurons
def model_builder(hp):
    model = models.Sequential()
    
    model.add(layers.Conv2D(28, (3,3), activation = "relu", data_format = 'channels_last', input_shape=()))
    model.add(layers.MaxPooling((2,2)))    
    
    model.add(layers.Conv2D(56, (3,3), activation = "relu", data_format = 'channels_last', input_shape=()))
    model.add(layers.MaxPooling((2,2)))    

    model.add(layers.Flatten())
    
    # Dense
    
    model.add(layers.Dense(hp.Int('units', min_value = 16, max_value = 128, step=16), activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.compile(optimizer=keras.optiizers.legacy.Adam(learning_rate = hp.Choice('learning_rate', values = [.01,.001,.0001])), loss = 'binary_crossentropy', )
    return model
#5) 2 fully connected layers

#6) train the model
