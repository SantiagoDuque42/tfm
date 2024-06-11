import numpy as np
import utils
import tensorflow as tf
import model_loading
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

class Modelo_Ensemble():
    
    def __init__(self):
        self.loss_object, self.optimizer = utils.get_model_objects()
      
    
    def generar_modelos_sombra(self, num_shadow_models, tipo_modelo):   
        """Función de generación de los modelos que se emplearán para generar los ataques adversariales que servirán como
           datos de entrenamiento del modelo ensemble final. """
        
        shadow_models = []
        
        if(tipo_modelo == 'pgd'):
            shadow_models = [model_loading.cargar_pgd_model() for _ in range(num_shadow_models)]
        elif(tipo_modelo == 'naive'):
            shadow_models = [model_loading.cargar_naive_model() for _ in range(num_shadow_models)]
        else:
            shadow_models = [model_loading.cargar_baseline_model() for _ in range(num_shadow_models)]
        
        shadow_classifiers = []
        for i in range(num_shadow_models):
            classifier = TensorFlowV2Classifier(model=shadow_models[i], 
                                                nb_classes=2, 
                                                input_shape=(50,50,3), 
                                                loss_object=self.loss_object,
                                                clip_values=(0, 1), 
                                                channels_first=False)
            shadow_classifiers.append(classifier) 
            
        return shadow_classifiers
            
    def generar_muestras_adversariales(self, x_train, y_train, shadow_classifiers, tipo_ataque):
        """Función para generar muestras adversariales a partir de los clasificadores sombra generados, y según
           el método especificado en los parámetros."""
        adv_x = []
        adv_y = []
        attack = None
        for i in range(len(x_train)):
            for model in shadow_classifiers:        
                if(tipo_ataque=="FGSM" or tipo_ataque=="fgsm"):
                    attack = FastGradientMethod(estimator=model, eps=0.15)
                elif(tipo_ataque=="CW" or tipo_ataque=="cw"):
                    attack = CarliniLInfMethod(classifier=model,
                                          max_iter=5,
                                          learning_rate=0.01,
                                          initial_const=1e0,
                                          largest_const=15e-1)

                adv_x.append(attack.generate(x_train[i:i+1]))
                adv_y.append(y_train[i:i+1])

        return tf.concat(adv_x, axis=0), tf.concat(adv_y, axis=0)  # Repeat labels for consistency
    
    def generar_datos(self, train_data, train_labels, num_muestras, tipo_ataque, tipo_modelo, num_shadow_models):
        """Función para generar los datos de entrenamiento, combinando las muestras adversariales a partir de los modelos
           establecidos con imágenes originales, según el tipo de ataque adversarial"""
        
        shadow_classifiers = self.generar_modelos_sombra(num_shadow_models, tipo_modelo)

        train_data = train_data[:num_muestras]
        train_labels = train_labels[:num_muestras]

        # Tipo de ataque mezclado, i.e. combinando dos métodos, FGSM y Carlini-Wagner
        if(tipo_ataque=="Mixed" or tipo_ataque=="mixed" or tipo_ataque=="MIXED"):
            half_len_data = int(len(train_data)/2)

            adv_x_train_1, adv_y_train_1 = self.generar_muestras_adversariales(
                train_data[:half_len_data], train_labels[:half_len_data], shadow_classifiers, "FGSM")

            adv_x_train_2, adv_y_train_2 = self.generar_muestras_adversariales(
                train_data[half_len_data:], train_labels[half_len_data:], shadow_classifiers, "CW")

            adv_x_train = np.concatenate((adv_x_train_1, adv_x_train_2), axis=0)
            adv_y_train = np.concatenate((adv_y_train_1, adv_y_train_2), axis=0)
            
        # Tipo de ataque singular, i.e. o bien FGSM o bien CW
        else:
            adv_x_train, adv_y_train = self.generar_muestras_adversariales(train_data, 
                                                                     train_labels, 
                                                                     shadow_classifiers, 
                                                                     tipo_ataque)

        # print(len(adv_x_train))
        # print(len(adv_y_train))
        # print(len(train_data))
        # print(len(train_labels))

        adv_x_train = tf.cast(adv_x_train, tf.float64)


        joined_data = np.concatenate([adv_x_train, train_data[:(num_muestras*num_shadow_models)]], axis=0)
        joined_labels = np.concatenate([adv_y_train, train_labels[:(num_muestras*num_shadow_models)]], axis=0)


        return joined_data, joined_labels
    
    
    def generar_datos_entrenamiento(self, train_data, train_labels, num_muestras, 
                                    tipo_ataque='Mixed', 
                                    tipo_modelo='pgd', 
                                    num_shadow_models=3):
        """Función auxiliar de generación de datos de entrenamiento para facilitar legibilidad"""
        shadow_train_data, shadow_trained_labels = self.generar_datos(train_data, 
                                                                      train_labels, 
                                                                      num_muestras, 
                                                                      tipo_ataque, 
                                                                      tipo_modelo, 
                                                                      num_shadow_models)
        
        return shadow_train_data, shadow_trained_labels
        
        
    def entrenar_modelo_ensemble(self, epochs, train_data, train_labels, num_muestras, tipo_ataque, tipo_modelo, num_shadow_models):
        """Función para entrenar el modelo ensemble.
        Args:
        -train_data, train_labels: Imágenes y etiquetas de entrenamiento
        -epochs: Periodos de entrenamiento
        -num_muestras: Número de muestras adversariales a generar por clasificador sombra
        -tipo_ataque: Tipo de ataque adversarial ('fgsm', 'cw' o 'mixed')
        -tipo_modelo: Tipo de modelo con el que generar los ataques ('pgd', 'naive' o 'baseline')
        -num_shadow_models: Número de clasificadores sombra con los que generar las muestras adversariales de entrenamiento
        """
        train_data, train_labels = self.generar_datos_entrenamiento(train_data, 
                                                                    train_labels, 
                                                                    num_muestras, 
                                                                    tipo_ataque, 
                                                                    tipo_modelo, 
                                                                    num_shadow_models)
        
        ensemble_model = utils.create_baseline_model()
        ensemble_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        ensemble_model.fit(train_data, 
                           train_labels, 
                           epochs=epochs)
        
        return ensemble_model