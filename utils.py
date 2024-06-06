# ****************************************************

# IMPORTS

# ****************************************************

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from keras import layers
print(np.__version__)
print(tf.__version__)
print(keras.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
from tensorflow.keras import Model

# ****************************************************

# PARÁMETROS DE IMÁGEN Y MODELO

# ****************************************************

num_classes = 2
width = 50
height = 50
channels = 3 # RGB
img_size = (width, height)
img_shape = (width, height, channels)
batch_size = 32




# ****************************************************

# FUNCIONES AUXILIARES

# ****************************************************



def shuffle_together(train_data, train_labels):
    """Mezcla dos datasets distintos, asegurándose de se mantiene la correspondencia 1:1 entre ambos"""
    
    num_samples = train_data.shape[0]

    shuffled_indices = np.random.permutation(num_samples)

    shuffled_data = train_data[shuffled_indices]
    shuffled_labels = train_labels[shuffled_indices]

    return shuffled_data, shuffled_labels


def convert_to_single_label(one_hot_labels):
    """Convierte la etiqueta completa (e.g.: [0. 1.]) en simple (e.g.: 1)"""
    return np.argmax(one_hot_labels, axis=1)



# ****************************************************

# OBJETOS DE MODELO

# ****************************************************

def get_loss_and_accuracy():
    """Devuelve métricas empleadas para monitorear algunos entrenamientos, como el adversarial con PGD"""
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.BinaryAccuracy(name='test_accuracy')
    
    return train_loss, train_accuracy, test_loss, test_accuracy


def get_model_objects():
    """Devuelve objetos necesarios para crear un TensorFlowV2Classifier, entre otros"""
    loss_object = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam()
    
    return loss_object, optimizer
    
    


# ****************************************************

# FUNCION DE GENERACIÓN DE MODELO BASE

# ****************************************************


def create_baseline_model():
    """Devuelve el modelo base sin defensas adversariales"""

    # Modelo secuencial de keras
    model = Sequential()

    # Tres capas convolucionales
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Capa flatten
    model.add(Flatten())

    # Dos capas densas con dropout del 15%
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))  

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.15))  

    # Capa de salida con el número de clases
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def create_baseline_model_sigmoid():

    # Modelo secuencial de keras
    model = Sequential()

    # Tres capas convolucionales
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Capa flatten
    model.add(Flatten())

    # Dos capas densas con dropout del 15%
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))  

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.15))  

    # Capa de salida con el número de clases
    model.add(Dense(1, activation='sigmoid'))
    
    return model
