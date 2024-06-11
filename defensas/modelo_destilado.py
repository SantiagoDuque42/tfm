import numpy as np
import utils
import tensorflow
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class Destilador(keras.Model):
    """Clase interna del Destilador. Define dos modelos, el profesor y el estudiante, que destilará la información del profesor, así como la 
        función custom para definir la pérdida con ambos modelos"""
    
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        perdida_estudiante,
        perdida_profesor,
        alpha=0.1,
        temperatura=3,
    ):
        """Override de la función tf.keras.Model.compile, para proporcionar nuevos parámetros (pérdidas custom, alpha y temperatura)"""

        super().compile(optimizer=optimizer, metrics=metrics)
        self.perdida_estudiante = perdida_estudiante
        self.perdida_profesor = perdida_profesor
        self.alpha = alpha
        self.temperatura = temperatura

        
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        """Override de la función tf.keras.Model.compute_loss, para definir nuestras pérdidas de modelo profesor y modelo estudiante."""
           
        # Obtiene las predicciones del modelo del profesor
        prediccion_profesor = self.teacher(x, training=False)
        
        # Calcula la pérdida del modelo del estudiante
        perdida_estudiante = self.perdida_estudiante(y, y_pred)

        # Calcula la pérdida de destilación, que es la diferencia entre las predicciones del profesor y del estudiante
        # Las predicciones se pasan a través de una función softmax y se escalan por la temperatura
        perdida_profesor = self.perdida_profesor(
            tensorflow.nn.softmax(prediccion_profesor / self.temperatura, axis=1),
            tensorflow.nn.softmax(y_pred / self.temperatura, axis=1),
        ) * (self.temperatura**2)

        # La pérdida total es una suma ponderada de la pérdida del estudiante y la pérdida de destilación
        loss = self.alpha * perdida_estudiante + (1 - self.alpha) * perdida_profesor
        
        # Devuelve la pérdida total
        return loss
    
    def call(self, x):
        return self.student(x)

class Modelo_Destilado():
    
    def __init__(self):
        
        # Modelo secuencial de keras
        self.modelo_profesor = Sequential([

            # Tres capas convolucionales
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50,50,3)),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Capa flatten
            layers.Flatten(),

            # Dos capas densas con dropout del 15%
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.15),

            layers.Dense(256, activation='relu'),
            layers.Dropout(0.15),

            # Capa de salida con el número de clases. No es necesaria la función de activación 'softmax',
            # pues se incluye en el cálculo de la pérdida en compute_loss()
            layers.Dense(2),
            ],
            name = "modelo_original_destilacion"
        )

        self.modelo_estudiante = Sequential([

            # Tres capas convolucionales
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50,50,3)),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Capa flatten
            layers.Flatten(),

            # Dos capas densas con dropout del 15%
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.15),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.15),

            # Capa de salida con el número de clases. No es necesaria la función de activación 'softmax',
            # pues se incluye en el cálculo de la pérdida en compute_loss()
            layers.Dense(2),
            ],
            name = "modelo_destilado"
        )
        
    def get_modelo(self):
        """Getter para el modelo combinado con el estudiante y el profesor"""
        destilador = Destilador(student=self.modelo_estudiante, teacher=self.modelo_profesor)
        destilador.compile(
            optimizer='adam',
            metrics=['accuracy'],
            perdida_estudiante=keras.losses.BinaryCrossentropy(from_logits=True),
            perdida_profesor=keras.losses.KLDivergence(),
            alpha=0.1,
            temperatura=5,
        )
        
        return destilador
    
    def entrenar_modelo_profesor(self, train_data, train_labels, epochs):
        self.modelo_profesor.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.modelo_profesor.fit(train_data, train_labels, epochs=epochs)
        
    def entrenar_modelo_estudiante(self, train_data, train_labels, epochs):
        self.entrenar_modelo_profesor(train_data, train_labels, epochs)
        
        destilador = self.get_modelo()

        # Destila el profesor al estudiante
        destilador.fit(train_data, train_labels, epochs=epochs)
        return destilador
    
    def entrenar_modelo_destilado(self, train_data, train_labels, epochs):
        
        modelo_destilado = self.entrenar_modelo_estudiante(train_data, train_labels, epochs)
        
        return modelo_destilado
      
