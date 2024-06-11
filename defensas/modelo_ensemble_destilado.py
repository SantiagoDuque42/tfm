import numpy as np
import utils
import tensorflow
import model_loading
import numpy as np
import ataques.ataques
import defensas.modelo_ensemble
import defensas.modelo_destilado
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

class Modelo_Ensemble_Destilado():
    
    def __init__(self):  
        self.loss_object, self.optimizer = utils.get_model_objects()
        self.modelo_ensemble = model_loading.cargar_ensemble_model()
        
        
        
    def generar_datos(self, data, labels):
        """Función para generar los datos de entrenamiento de ensemble, según la clase de modelo_ensemble.py"""
        ensemble = defensas.modelo_ensemble.Modelo_Ensemble()
        
        # Se generan los datos de entrenamiento a partir de la función detallada en la clase de defensa ensemble
        shadow_data, shadow_labels = ensemble.generar_datos_entrenamiento(data, labels, 5000, 'fgsm')
        
        return shadow_data, shadow_labels
        
        
    def get_modelo(self, data, labels):
        """Función para generar el modelo destilado a partir del modelo ensemble de la clase homónima, y compilarlo sin entrenarlo"""
        shadow_data, shadow_labels = self.generar_datos(data, labels)
        
        # Se genera el modelo destilador a partir de la clase homónima
        destilador = defensas.modelo_destilado.Modelo_Destilado()
        modelo_estudiante = destilador.modelo_estudiante
        
        # Se destila desde el modelo ensemble ya entrenado al nuevo modelo estudiante de la clase destilador
        modelo = defensas.modelo_destilado.Destilador(student=modelo_estudiante, teacher=self.modelo_ensemble)
        modelo.compile(
            optimizer='adam',
            metrics=['accuracy'],
            perdida_estudiante=keras.losses.BinaryCrossentropy(from_logits=True),
            perdida_profesor=keras.losses.KLDivergence(),
            alpha=0.1,
            temperatura=5,
        )
        
        print(modelo.dtype)
        
        return modelo#, shadow_data, shadow_labels
        
        
    def entrenar_modelo_ensemble_destilado(self, data, labels, epochs):    
        modelo, shadow_data, shadow_labels = self.get_modelo(data, labels)
        modelo.fit(shadow_data, shadow_labels, epochs=epochs)
        
        return modelo