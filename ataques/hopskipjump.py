from art.utils import to_categorical
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import HopSkipJump
import numpy as np
import model_loading
import utils
import random
 
class Ataque_hsj():
    def __init__(self, model):
        self.model = model
        
        self.loss_object, self.optimizer = utils.get_model_objects()

        self.classifier = TensorFlowV2Classifier(model=self.model, nb_classes=2, input_shape=(50,50,3), loss_object=self.loss_object,
                                        clip_values=(0, 1), channels_first=False)
              
        
        self.attack_hsj = HopSkipJump(classifier=self.classifier, targeted=False, max_iter=0, max_eval=500, init_eval=10)

    def generar_listas_imagenes(self, train_data, train_labels):
        """Función para generar dos listas de imágenes originales, una por cada etiqueta correspondiente, para generar
           a partir de ellas los ataques HSJ targeted
        """
        negative_imgs_list = []
        positive_imgs_list = []
        for i in range(len(train_data)):
            label = np.argmax(train_labels[i], axis=0)
            if (label == 0.0):
                negative_imgs_list.append((train_data[i]))
            elif (label == 1.0):
                positive_imgs_list.append((train_data[i]))

        positive_imgs_list = np.asarray(positive_imgs_list)
        negative_imgs_list = np.asarray(negative_imgs_list)   

        return positive_imgs_list, negative_imgs_list
    
    def generar_hsj_negativo(self, data, negative_imgs_list, negative_label, num_imgs, num_iter, iter_step, x_adv):
        """Función para generar ataques adversariales HSJ desde imágenes con etiqueta 0"""
        
        adv_data_hsj_negative = []
        for i in range(len(negative_imgs_list)):
            # Expande las dimensiones de la imagen para estandarizarla a (None, 50, 50, 3)
            img = np.expand_dims(negative_imgs_list[i], axis=0)
            for _ in range(num_iter):       
                # Genera la imagen adversaria usando el método HopSkipJump
                x_adv = self.attack_hsj.generate(x=img, y=negative_label, x_adv_init=x_adv, resume=True)
                
                # Establece el número máximo de iteraciones
                self.attack_hsj.max_iter = iter_step       
            adv_data_hsj_negative.append(x_adv)
        return adv_data_hsj_negative

    def generar_hsj_positivo(self, data, positive_imgs_list, positive_label, num_imgs, num_iter, iter_step, x_adv):
        """Función para generar ataques adversariales HSJ desde imágenes con etiqueta 1"""
        
        adv_data_hsj_positive = []
        for i in range(len(positive_imgs_list)): 
            # Expande las dimensiones de la imagen para estandarizarla a (None, 50, 50, 3)
            img = np.expand_dims(positive_imgs_list[i], axis=0)
            for _ in range(num_iter):    
                # Genera la imagen adversaria usando el método HopSkipJump
                x_adv = self.attack_hsj.generate(x=img, y=positive_label, x_adv_init=x_adv, resume=True)
                
                 # Establece el número máximo de iteraciones 
                self.attack_hsj.max_iter = iter_step       
            adv_data_hsj_positive.append(x_adv)
        return adv_data_hsj_positive
    
    
    
    
    def generar_ataque_hsj(self, train_data, train_labels, num_imgs, num_iter, iter_step):
        """Función para generar ataques HSJ combinados, a partir de las listas de imágenes originales con etiqueta 0 y 1"""
                 
        # Genera listas de imágenes positivas y negativas    
        positive_imgs_list, negative_imgs_list = self.generar_listas_imagenes(train_data, train_labels)
        
        # Elige una imagen aleatoria de la lista de imágenes positivas
        clean_img = random.choice(positive_imgs_list)
        
        # Expande las dimensiones de la imagen adversarial para estandarizarla a (None, 50, 50, 3)
        #clean_img = np.expand_dims(clean_img, axis=0)
        clean_img = np.reshape(clean_img, (1, *clean_img.shape))
        
        # Genera imágenes adversariales con etiquetas positivas
        adv_data_hsj_positive = self.generar_hsj_positivo(train_data, 
                                                           positive_imgs_list, 
                                                           np.asarray([0.0, 1.0]), 
                                                           num_imgs, 
                                                           num_iter, 
                                                           iter_step,
                                                           clean_img)
        
        # Genera imágenes adversariales con etiquetas negativas
        adv_data_hsj_negative = self.generar_hsj_negativo(train_data, 
                                                           negative_imgs_list, 
                                                           np.asarray([1.0, 0.0]), 
                                                           num_imgs, 
                                                           num_iter, 
                                                           iter_step,
                                                           clean_img)
  

        # Convierte las listas de imágenes adversariales en arrays numpy
        adv_data_hsj_positive = np.asarray(adv_data_hsj_positive)
        adv_data_hsj_negative = np.asarray(adv_data_hsj_negative)
        
        # Redimensiona los arrays de imágenes adversariales
        #np.reshape(adv_data_hsj_positive, (50,50,3)) 
        #np.reshape(adv_data_hsj_negative, (50,50,3)) 
        
         # Combina los arrays de imágenes adversariales
        adv_data_hsj_combined = np.concatenate((adv_data_hsj_positive, adv_data_hsj_negative), axis=0)
        adv_data_hsj_combined = np.squeeze(adv_data_hsj_combined, axis=1)
        
        # Devuelve el array combinado de imágenes adversariales, así como los tamaños de cada lista de imágenes para
        # facilitar el benchmarking
        return adv_data_hsj_combined, len(adv_data_hsj_positive), len(adv_data_hsj_negative)