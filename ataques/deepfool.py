import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow import GradientTape
from tensorflow.keras import Model
import copy

class Ataque_DeepFool():
    
    def deepfool(self, img, modelo, num_clases=2, overshoot=0.02, max_iter=50, shape=(50, 50, 3)):
        """Genera una imagen adversarial mediante DeepFool y acorde con una imagen original y
        un modelo de clasificación de imagen.
        Argumentos: 
            -img: Una imagen como numpy array con forma=shape
            -modelo: El modelo que intentará engañar el ataque
            -num_classes: Número de clases para la clasificación
            -overshoot: Margen que tiene la perturbación para exceder el límite
            -max_iter: Número de iteraciones máximas por ataque
            -shape: Forma de la imagen, en nuestro caso 50x50 píxeles con 3 canales RGB (50,50,3)"""

        # Convierte la imagen a float32 y la convierte en una matriz numpy
        imagen = tf.cast(img, tf.float32)
        imagen = imagen.numpy()

        # Obtiene la predicción del modelo para la imagen y aplana la salida
        etiquetas_pred_array = modelo(imagen).numpy().flatten()
        
         # Obtiene los índices que ordenarían la matriz de predicción en orden descendente
        etiquetas_pred_flatten = (np.array(etiquetas_pred_array)).flatten().argsort()[::-1]
        
        # Obtiene la etiqueta con la predicción más alta
        etiquetas_pred = etiquetas_pred_flatten[0]
        imagen_adv = copy.deepcopy(imagen)

        # Inicializa la perturbación y la perturbación total a cero
        w = np.zeros(shape)
        proyeccion = np.zeros(shape)

         # Inicializa el contador del bucle y la etiqueta predicha
        num_loop = 0
        x = tf.Variable(imagen_adv)
        adv_etiquetas = modelo(x)
        etiqueta_final = etiquetas_pred

        # Define la función de pérdida
        def perdida(logits, pred_etiquetas, i):
            return logits[0, pred_etiquetas[i]]

         # Mientras la etiqueta predicha sea la misma que la etiqueta original y el número de iteraciones sea
         # menor que el máximo
        while etiqueta_final == etiquetas_pred and num_loop < max_iter:

            # Inicializa la perturbación a infinito (norma L_inf)
            norma_l_inf = np.inf
            
            # Crea un vector codificado en one-hot para la etiqueta original
            one_hot_etiquetas_0 = tf.one_hot(etiquetas_pred, num_clases)

            # Calcula el gradiente de la pérdida con respecto a la imagen de entrada
            with tf.GradientTape() as tape:
                tape.watch(x)
                adv_etiquetas = modelo(x)
                valor_perdida = perdida(adv_etiquetas, etiquetas_pred_flatten, 0)
            gradiente_orig = tape.gradient(valor_perdida, x)

            # Para cada clase
            for k in range(1, num_clases):
                # Crea un vector codificado en one-hot para la clase actual
                one_hot_etiquetas_k = tf.one_hot(etiquetas_pred_flatten[k], num_clases)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    adv_etiquetas = modelo(x)
                    valor_perdida = perdida(adv_etiquetas, etiquetas_pred_flatten, k)
                # Calcula el gradiente de la pérdida con respecto a la imagen de entrada
                gradiente_interna = tape.gradient(valor_perdida, x)

                # Calcula la diferencia entre el gradiente actual y el gradiente original
                gradiente_total = gradiente_interna - gradiente_orig

                 # Calcula la diferencia entre la puntuación de la clase actual y 
                 # la puntuación de la clase original
                diff_etiquetas = (adv_etiquetas[0, etiquetas_pred_flatten[k]] - adv_etiquetas[0, etiquetas_pred_flatten[0]]).numpy()

                # Calcula la perturbación para la clase actual
                norma_k = abs(diff_etiquetas) / np.linalg.norm(tf.reshape(gradiente_total, [-1]))

                # Si la perturbación actual es menor que la mínima encontrada hasta ahora, 
                # actualiza la perturbación mínima y el gradiente correspondiente
                if (norma_k < norma_l_inf):
                    norma_l_inf = norma_k
                    w = gradiente_total

            # Calcula la perturbación para la iteración actual
            r_i = (norma_l_inf + 1e-4) * w / np.linalg.norm(w)
            # Añade la perturbación actual a la perturbación total
            proyeccion = np.float32(proyeccion + r_i)

            # Añade la perturbación total a la imagen original
            imagen_adv = imagen + (1 + overshoot) * proyeccion

            # Actualiza la imagen de entrada para la próxima iteración
            x = tf.Variable(imagen_adv)

            # Obtiene la predicción del modelo para la imagen perturbada
            adv_etiquetas = modelo(x)
            
            # Obtiene la etiqueta con la predicción más alta
            k_i = np.argmax(np.array(adv_etiquetas).flatten())

            # Incrementa el contador del bucle
            num_loop += 1

        # Calcula la perturbación final
        proyeccion = (1 + overshoot) * proyeccion

        # Devuelve la imagen perturbada
        return imagen_adv #, proyeccion, num_loop, etiquetas, etiquetas_pred


    def generar_ataque_deepfool(self, original_data, model, num_clases=2, overshoot=0.02, max_iter=50, shape=(50, 50, 3)):
        """Función para generar ataques DeepFool para un array de imágenes"""
        ds = []
        for img in original_data:
            imagen = np.expand_dims(img, axis=0)
            adv = self.deepfool(imagen, model, num_clases=num_clases, overshoot=overshoot, max_iter=max_iter, shape=shape)
            ds.append(adv)

        return np.array(ds, dtype='float64')