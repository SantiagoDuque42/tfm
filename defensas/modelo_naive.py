import model_loading
import ataques.ataques
import numpy as np
import copy
import utils

class Modelo_Naive():

    def __init__(self):
        pass
        
                 
    def generar_datos_entrenamiento_adv(self, train_data, train_labels, train_data_adv_fgsm, adv_num):
        
        # Las imágenes de entrenamiento se combinan de tal modo que se junten x imágenes originales y los x correspondiendes ataques, 1:1 por
        # cada imagen original. Las etiquetas, por tanto, están por duplicado para mantener la consistencia.          
        train_data_combined = np.concatenate((train_data[:adv_num], train_data_adv_fgsm[:adv_num]), axis=0)
        train_labels_combined = np.concatenate((train_labels[:adv_num], train_labels[:adv_num]), axis=0)

        copy_train_data = train_data_combined.copy()
        copy_train_labels = train_labels_combined.copy()

        shuffled_combined_train_data, shuffled_combined_train_labels = utils.shuffle_together(copy_train_data, copy_train_labels)
        
        return shuffled_combined_train_data, shuffled_combined_train_labels
        

    def entrenar_modelo_naive(self, modelo, train_data, train_labels, adv_ratio, epochs):
        # La tasa de datos de entrenamiento que se va a atacar e incorporar al entrenamiento. E.g. Si el adv_ratio es 0.5, se usará la mitad
        # de los datos de entrenamiento, así como los ataques adversariales correspondientes a esa misma mitad.
        adv_num = int(adv_ratio*len(train_data))
        
        fgsm = ataques.ataques.Ataque_fgsm(modelo)
        train_data_adv_fgsm = fgsm.generar_ataque_fgsm(train_data, 0.25)
        
        modelo = utils.create_baseline_model()
        
        modelo.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
        data, labels = self.generar_datos_entrenamiento_adv(train_data, train_labels, train_data_adv_fgsm, adv_num)

        modelo.fit(data, labels, epochs=epochs, batch_size=32)
        
        return modelo