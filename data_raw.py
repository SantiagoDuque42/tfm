import utils
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from os import listdir
import tensorflow as tf

# ****************************************************

# CARGAR DATASET

# ****************************************************

dataset_path = 'C:/Users/santi/Desktop/MASTER/Modulos/TFM/Datasets/kaggle_breast_histopathology_dataset'
dest_folder = 'processed_dataset'
#os.makedirs(dest_folder, exist_ok=True)
                                
def extraer_dataset(path_origen, path_destino):
    """Función para extraer el dataset de las carpetas originales y organizarlo por categorías en un único directorio.
       Args:
       path: directorio donde se encuentra el dataset original
       path_destino: directorio donde se va a copiar organizado el dataset
    """
    
    # La estructura de las carpetas originales del dataset es:
    # 1234 (nº paciente)
        # ~ 0 (carpeta de etiqueta, en este caso negativa)
            # ~ 1234_idx5_x101_y201_class0.png (nombre de la imagen concreta, con el número de paciente, coordenadas y categoría)
            
    # De este modo, se loopea primero por las carpetas de pacientes; dentro de ellas por las carpetas de cada categoría, y dentro
    # de ellas por las imágenes, y las copia con 'shutil' a la dirección establecida en el segundo parámetro
    for folder in os.listdir(path_origen):
        folder_path = os.path.join(path_origen, folder)
        
        for category in os.listdir(folder_path):
            category_label = category
            category_path = os.path_origen.join(folder_path, category)
            
            for filename in os.listdir(category_path):
                image_path = os.path_origen.join(category_path, filename)
                if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"File is not an image: {image_path}")
                    continue  
                        
                destination_path = os.path_origen.join(path_destino, category_label)                    
                destination_file = os.path_origen.join(destination_path, filename)
                shutil.copy(image_path, destination_file)   
                
                
limite_imagenes = 40000
#path_dataset_original = 'C:/Users/santi/Desktop/MASTER/Modulos/TFM/Datasets/kaggle_breast_histopathology_dataset'

# ****************************************************

# PRE-PROCESAMIENTO DEL DATASET

# ****************************************************

def procesar_datasets(limite_imagenes, path_dataset_original):
    """Función para cargar las imágenes a variables y realizar el split train/test/validación"""
    procesar_data(ds_0, ds_1, path_dataset_original, limite_imagenes)

    dataset_0, dataset_1, dataset_0_etiquetas, dataset_1_etiquetas = cargar_imgs(ds_0, ds_1)

    train_data, train_etiquetas, test_data, test_etiquetas, val_data, val_etiquetas = split_dataset(
     dataset_0, dataset_1, dataset_0_etiquetas, dataset_1_etiquetas, 0.55, 0.65)
    
    return train_data, train_etiquetas, test_data, test_etiquetas, val_data, val_etiquetas


def get_train_and_test_data():
    """Función para adaptar a la shape correcta los tataset de etiquetas y mezclar los datasets"""
    train_etiquetas = utils.to_categorical(train_etiquetas, 2)
    test_etiquetas = utils.to_categorical(test_etiquetas, 2)
    val_etiquetas = utils.to_categorical(val_etiquetas, 2)


    shuffled_train_data, shuffled_train_etiquetas = utils.shuffle_together(train_data, train_etiquetas)
    shuffled_test_data, shuffled_test_etiquetas = utils.shuffle_together(test_data, test_etiquetas)
    
    return shuffled_train_data, shuffled_train_etiquetas, shuffled_test_data, shuffled_test_etiquetas


# Importamos el dataset a una variable

#path_dataset_completo = 'C:/Users/santi/Desktop/MASTER/Modulos/TFM/Notebooks/full_processed_dataset/'

def procesar_data(dataset_0, dataset_1, path_dataset_completo, path, limite):
    """Función para dividir las imágenes en distintos datasets según su categoría, con opcional límite para facilidad de procesamiento"""
    archivos = listdir(path_dataset_completo)
    for i, categoria in enumerate(archivos):
        path_clase = f'{path}{i}/'
        subarchivos = listdir(path_clase)
        dataset = dataset_0 if categoria == '0' else dataset_1
        for n, subarchivo in enumerate(subarchivos):
            if n > limite:
                print(f"limite de imágenes '{categoria}' alcanzado")
                break
            path_imagen = f'{path_clase}{subarchivo}'
            dataset.append([path_imagen, categoria])

def cargar_imgs(dataset_0, dataset_1, dims_imagen):
    """Función para cargar las imágenes desde el archivo a una variable np.ndarray legible, usando PIL.Image"""
    def procesar_dataset(dataset):
        """Función auxiliar que loopea a través de los datasets y va abriendo las imágenes"""
        data = [img for img, _ in dataset]
        etiquetas = [etiqueta for _, etiqueta in dataset]
        resized = []
        for path_imagen in data:
            # Se abren las imágenes y se resizean por si hay alguna con el tamaño uncorrecto, usando Lanczos para asegurar que se 
            # mantiene la calidad.
            imagen = Image.open(path_imagen)
            resized_imagen = imagen.resize(dims_imagen, Image.LANCZOS)
            resized.append(resized_imagen)
            
        # Se pasan a np.ndarray y se shufflean
        np_imagenes = np.array([np.array(imagen) / 255.0 for imagen in resized])
        np_imagenes = shuffle(np_imagenes, random_state=42)
        etiquetas = shuffle(etiquetas, random_state=42)
        #print(f'dataset_{etiquetas[0]} shape : {np_imagenes.shape}')
        return np_images, etiquetas

    dataset_0, dataset_0_etiquetas = procesar_dataset(dataset_0)
    dataset_1, dataset_1_etiquetas = procesar_dataset(dataset_1)

    return dataset_0, dataset_1, dataset_0_etiquetas, dataset_1_etiquetas
    

def split_dataset(dataset_0, dataset_1, dataset_0_etiquetas, dataset_1_etiquetas, train_split, val_split):
    """Función para realizar los splits train/test/validación"""
    
    def split_data(dataset, etiquetas, train_split, val_split):
        """Función auxiliar para reusar para las dos categorías distintas"""
        train_data, temp_data, train_etiquetas, temp_etiquetas = train_test_split(
            dataset, etiquetas, test_size=1.0 - train_split, stratify=etiquetas, random_state=42
        )
        val_data, test_data, val_etiquetas, test_etiquetas = train_test_split(
            temp_data, temp_etiquetas, test_size=1.0 - val_split, stratify=temp_etiquetas, random_state=42
        )
        return train_data, val_data, test_data, train_etiquetas, val_etiquetas, test_etiquetas

    train_data_0, val_data_0, test_data_0, train_etiquetas_0, val_etiquetas_0, test_etiquetas_0 = split_data(dataset_0, 
                                                                                                    dataset_0_etiquetas, 
                                                                                                    train_split, 
                                                                                                    val_split)
    train_data_1, val_data_1, test_data_1, train_etiquetas_1, val_etiquetas_1, test_etiquetas_1 = split_data(dataset_1, 
                                                                                                    dataset_1_etiquetas, 
                                                                                                    train_split, 
                                                                                                    val_split)

    train_data = np.concatenate((train_data_0, train_data_1), axis=0)
    train_etiquetas = np.concatenate((train_etiquetas_0, train_etiquetas_1), axis=0)

    test_data = np.concatenate((test_data_0, test_data_1), axis=0)
    test_etiquetas = np.concatenate((test_etiquetas_0, test_etiquetas_1), axis=0)

    val_data = np.concatenate((val_data_0, val_data_1), axis=0)
    val_etiquetas = np.concatenate((val_etiquetas_0, val_etiquetas_1), axis=0)
    
    return train_data, train_etiquetas, test_data, test_etiquetas, val_data, val_etiquetas