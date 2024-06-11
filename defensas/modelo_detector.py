import tensorflow as tf
import numpy as np
import utils
from ataques import ataques
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from art.estimators.classification import TensorFlowV2Classifier

class Modelo_Detector():
    
    def __init__(self):
        self.modelo_clasificador = None
        self.modelo_detector = None
        
        
    def generar_dataset_combinado(self, modelo_atacante, train_data, train_labels):
        """Función para generar el dataset de entrenamiento que combina las imágenes originales con sus 
           correspondientes ataques adversariales, para el entrenamiento del detector"""
        fgsm = ataques.Ataque_fgsm(modelo_atacante, 0.2)
        train_data_adv_fgsm = fgsm.generar_ataque_fgsm(train_data)

        train_data_combinada = np.concatenate((train_data, train_data_adv_fgsm), axis=0)

        # Las imágenes originales se etiquetan con categoría 0, independientemente de su categoría original (indicios de cáncer)
        # Las imágenes adversariales se etiquetan con categoría 1, independientemente de su categoría original
        train_labels_adv_orig = np.full((len(train_data), 2), [1.0, 0.0])
        train_labels_adv = np.full((len(train_data), 2), [0.0, 1.0])

        train_etiquetas_combinada = [train_labels_adv_orig, train_labels_adv]
        train_etiquetas_combinada = np.array(np.concatenate(train_etiquetas_combinada, axis=0))
        
        copy_train_data = train_data_combinada.copy()
        copy_train_labels = train_etiquetas_combinada.copy()

        combined_train_data, combined_train_labels = utils.shuffle_together(copy_train_data, copy_train_labels)
        
        return combined_train_data, combined_train_labels

        
    def generar_modelos(self):
        """Función para generar las dos subredes"""
        
        # Capas convolucionales compartidas
        modelo_original = Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50,50,3)),
            tf.keras.layers.MaxPooling2D((2, 2))  
        ])


        # Sub-red 1: Clasificación binaria
        sub1_output = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(modelo_original.output)
        sub1_output = tf.keras.layers.MaxPooling2D((2, 2))(sub1_output)
        sub1_output = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(sub1_output)
        sub1_output = tf.keras.layers.MaxPooling2D((2, 2))(sub1_output)
        sub1_output = tf.keras.layers.Flatten()(sub1_output)
        sub1_output = tf.keras.layers.Dense(256, activation='relu')(sub1_output)
        sub1_output = tf.keras.layers.Dropout(0.25)(sub1_output)
        sub1_output = tf.keras.layers.Dense(128, activation='relu')(sub1_output)
        sub1_output = tf.keras.layers.Dropout(0.35)(sub1_output)
        sub1_output = tf.keras.layers.Dense(2, activation='softmax')(sub1_output)

        # Sub-red 2: Detección adversarial
        sub2_output = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(modelo_original.output)
        sub2_output = tf.keras.layers.MaxPooling2D((2, 2))(sub2_output)
        sub2_output = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(sub2_output)
        sub2_output = tf.keras.layers.MaxPooling2D((2, 2))(sub2_output)
        sub2_output = tf.keras.layers.Flatten()(sub2_output)
        sub2_output = tf.keras.layers.Dense(256, activation='relu')(sub2_output)
        sub2_output = tf.keras.layers.Dropout(0.15)(sub2_output)
        sub2_output = tf.keras.layers.Dense(128, activation='relu')(sub2_output)
        sub2_output = tf.keras.layers.Dropout(0.2)(sub2_output)
        sub2_output = tf.keras.layers.Dense(2, activation='softmax')(sub2_output)

        # Modelos separados
        self.modelo_clasificador = Model(inputs=modelo_original.input, outputs=sub1_output)
        self.modelo_detector = Model(inputs=modelo_original.input, outputs=sub2_output)
        
        return modelo_original
        
        
    def get_modelo(self):
        """Getter para el modelo combinado con el clasificador y el detector"""
        modelo = self.generar_modelos()
        self.modelo_clasificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.modelo_detector.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        modelo_detector_combinado = Model(inputs = modelo.input, 
                           outputs = [self.modelo_clasificador.output, self.modelo_detector.output])
        modelo_detector_combinado.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return modelo_detector_combinado

    def entrenar_modelo_clasificador(self, data, labels, epochs):
        self.modelo_clasificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.modelo_clasificador.fit(data, 
                                     labels, 
                                     epochs=epochs)
        
    def entrenar_modelo_detector(self, combined_train_data, combined_train_labels, epochs_detector):        
        self.modelo_detector.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.modelo_detector.fit(combined_train_data, 
                                 combined_train_labels, 
                                 epochs=epochs_detector)
        
    def entrenar_modelo_combinado(self, modelo_atacante, data, labels, epochs, epochs_detector):
        modelo = self.generar_modelos()
        combined_train_data, combined_train_labels = self.generar_dataset_combinado(modelo_atacante, data, labels)
        self.entrenar_modelo_clasificador(data, labels, epochs)
        self.entrenar_modelo_detector(combined_train_data, combined_train_labels, epochs_detector)
        
        # El modelo final es una combinación de las capas compartidas entre ambas subredes y sus dos salidas
        modelo_detector_combinado = Model(inputs = modelo.input, 
                           outputs = [self.modelo_clasificador.output, self.modelo_detector.output])
        
        return modelo_detector_combinado