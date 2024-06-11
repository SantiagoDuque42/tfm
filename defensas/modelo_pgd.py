import tensorflow as tf
from tensorflow import sign
from tensorflow import clip_by_value
from tensorflow import keras
from tensorflow import GradientTape
import utils
import numpy as np
from keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class Modelo_PGD():
    
    def __init__(self):    
        # Función de pérdida y optimizador
        self.loss_object = keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.Adam()

        # Métricas de monitoreo
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

    def crear_patron_adversarial(self, imagen, etiqueta, model, funcion_perdida): 
        # Comprobación de la shape correcta de imágenes y etiquetas
        if(imagen.shape == (50,50,3)):
            imagen = tf.expand_dims(imagen, axis=0)
        if(etiqueta.shape == (2,)):
            etiqueta = tf.expand_dims(etiqueta, axis=0)
        with tf.GradientTape() as tape:
            tape.watch(imagen)
            prediccion = model(imagen)
            perdida = funcion_perdida(etiqueta, prediccion)

        # Genera las gradientes de la pérdida con respecto a la imagen
        gradiente = tape.gradient(perdida, imagen)
        # Utiliza el signo de la gradiente para la perturbación
        gradiente_signo = tf.sign(gradiente)
        return gradiente_signo

    def generar_ataque_pgd(self, imagen, etiqueta, model, perdida, epsilon, num_pasos, alpha):
        adv_x = imagen
        for i in range(num_pasos):
            perturbaciones = self.crear_patron_adversarial(adv_x, etiqueta, model, perdida)
            
            # Se genera la imagen adversarial añadiendo la perturbación, el alpha y se normaliza  
            adv_x = adv_x + alpha*perturbaciones
            adv_x = tf.clip_by_value(adv_x, imagen - epsilon, imagen + epsilon)
            adv_x = tf.clip_by_value(adv_x, 0, 1)
        return adv_x

    def generar_imagenes_adversariales(self, imagenes, etiquetas, modelo, perdida, epsilon, num_pasos, alpha):
        adv = []
        for i in range(len(imagenes)):
            adv_img = self.generar_ataque_pgd(imagenes[i], etiquetas[i], modelo, perdida, epsilon, num_pasos, alpha)
            adv.append(adv_img)       
        return adv
    
    @tf.function
    def train_step(self, modelo, imagenes, etiquetas, random_start, epsilon, num_pasos, tamano_paso):
        """Función del paso de entrenamiento para el entrenamiento adversarial con PGD. Genera el ataque adversarial
        y actualiza manualmente las gradientes. 
        Parámetros:
        -modelo: KerasModel, Modelo a ser atacado
        -imagenes: np.ndarray, (n,50,50,3)
        -etiquetas: np.ndarray, (n,2)
        -random_start: Bool, determina si la perturbación con PGD se inicia de manera aleatoria
        -epsilon: Float, margen de amplitud de la perturbación
        -num_pasos: Int, número de iteraciones en la generación de la perturbación
        -tamano_paso: Int, multiplicador de magnitud de la gradiente del ataque
        """

        objeto_perdida, optimizador = utils.get_model_objects()
        perdida = keras.losses.BinaryCrossentropy()
        
        # Generamos las imágenes adversariales usando PGD
        #imagenes_adv = self.generar_ataque_pgd(modelo, imagenes, etiquetas, random_start, epsilon, num_pasos, tamano_paso)
        imagenes_adv = self.generar_imagenes_adversariales(imagenes, etiquetas, modelo, objeto_perdida, epsilon, num_pasos, tamano_paso)
        

        with tf.GradientTape() as tape:

            # Calculamos las pérdidas, original y adversarial, mediante un forward pass
            predicciones_original = modelo(imagenes, training=True)
            #predicciones_adversarial = modelo(imagenes_adv, training=True)

            perdida_original = self.loss_object(etiquetas, predicciones_original)
            #perdida_adversarial = self.loss_object(etiquetas, predicciones_adversarial)

            perdida_total = perdida_original # perdida_adversarial

        # Calculamos las gradientes
        gradientes = tape.gradient(perdida_total, modelo.trainable_variables)

        # Y las aplicamos (backpropagation)
        self.optimizer.apply_gradients(zip(gradientes, modelo.trainable_variables))

        # Actualizamos nuestras métricas
        self.train_loss(perdida_total)
        self.train_accuracy(etiquetas, predicciones_original)

        return predicciones_original
    
    
    def entrenar_modelo_pgd(self, train_data, train_labels, random_start, epsilon, max_num_training_steps, num_adv_steps, step_size):
        """Función para aplicar el entrenamiento adversarial con PGD a un modelo con los datos provistos en los parámetros."""
                
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(50000).batch(32)
        
        pgd_model = utils.create_baseline_model()

        pgd_model.compile(optimizer='adam', 
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        pgd_model.summary()
        
        for epoch in range(max_num_training_steps):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for images, labels in train_ds:   
                prediccions = self.train_step(pgd_model, 
                                              images, 
                                              labels, 
                                              random_start, 
                                              epsilon,
                                              num_adv_steps, 
                                              step_size)

            print(f'Época {epoch + 1}, Pérdida: {self.train_loss.result()}, Precisión: {self.train_accuracy.result()}')
            
        return pgd_model