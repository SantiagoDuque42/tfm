import utils
import numpy as np
import seaborn as sns
import art.metrics.metrics
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from art.estimators.classification import TensorFlowV2Classifier


loss_object, optimizer = utils.get_model_objects()
train_loss, train_accuracy, test_loss, test_accuracy = utils.get_loss_and_accuracy()

num_classes = 2
img_shape = (50,50,3)


def evaluate_detection_network(test_dataset, test_labels, model):
    """Función para evaluar el modelo de defensa con subred detectora"""
    correct_guesses = 0
    adv_images = 0
    non_adv_images = 0
    for i in range(len(test_dataset)):
        img = (np.expand_dims(test_dataset[i], axis=0))
        
        # e.g. ([0.62, 0.38], [0.23, 0.77]), 1er elemento es clasificacion y el 2º es la deteccion adversarial
        pred = model(img) 
        
        # Si se determina que la imagen es adversarial
        if(float(np.argmax(pred[1])) == 1.0):
            adv_images = adv_images+1
        else:   
            non_adv_images = non_adv_images+1
            if(np.argmax(pred[0]) == np.argmax(test_labels[i])):
                correct_guesses = correct_guesses+1
    
    # Calcula la decibilidad adversarial, es decir, la precisión a la hora de detectar imágenes adversariales
    decidibilidad = adv_images*100 / len(test_dataset)
    print("Decidibilidad Adversarial: %f" % decidibilidad)
    
    return decidibilidad

def analizar_precision_perturbacion(model, nombre_modelo, labels_orig, data_adv, adv_size, is_detection_model=False):
    
    """Función para evaluar los ataques y defensas adversariales.
    Argumentos:
        -model: El modelo de defensa a evaluar
        -nombre_modelo: Su nombre (baseline, naïve, ensemble...)
        -labels_orig: Las etiquetas originales de las imágenes sin alterar
        -data_adv: Los ataques adversariales correspondientes a dichas etiquetas
        -adv_size: Número de imágenes a evaluar
        -is_detection_model: Determina si el modelo de defensa es un modelo con subred detectora (caso especial) o no
    """
    
    if(is_detection_model):
        evaluate_detection_network(data_adv, labels_orig, model)
    else: 
        predict_data = model.predict(data_adv[:adv_size])
        predict_labels = np.argmax(predict_data, axis=1) 

        # Calcular precisión
        accuracy = accuracy_score(utils.convert_to_single_label(labels_orig[:adv_size]), predict_labels)
        print(f'Accuracy: {accuracy:.2f}')

        # Calcular exactitud
        precision = precision_score(utils.convert_to_single_label(labels_orig[:adv_size]), predict_labels)
        print(f'Precision: {precision:.2f}')

        # Calcular recall
        recall = recall_score(utils.convert_to_single_label(labels_orig[:adv_size]), predict_labels)
        print(f'Recall: {recall:.2f}')

        # Calcular puntuación F1
        f1 = f1_score(utils.convert_to_single_label(labels_orig[:adv_size]), predict_labels)
        print(f'F1 Score: {f1:.2f}')
        
        labels_orig_digit = np.argmax(labels_orig, axis=1) 
        
        # Calcular matriz de confusión
        conf_matrix = confusion_matrix(labels_orig_digit[:adv_size], predict_labels) #val_predict_labels)
        f,ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
        plt.xlabel("Etiqueta Asignada")
        plt.ylabel("Etiqueta Original")
        plt.title("Matriz de Confusión de Modelo " + nombre_modelo)
        plt.show()
        
        # Calcular sensibilidad y especificidad
    
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn+fp)

        print(f'Sensitivity: {recall:.2f}')
        print(f'Specificity: {specificity:.2f}')
        
        print("****************************************")
    


def empirical_robustness(model, data, tipo_ataque, num_imgs):
    """Función para calcular la robustez empírica de un modelo de defensa, según un tipo de ataque.
    La funcionalidad core la provee la Adversarial Robustness Toolbox (A.R.T.) con art.metrics.metrics.empirical_robustness()"""
                         
    test_classifier = TensorFlowV2Classifier(model=model, nb_classes=num_classes, input_shape=img_shape, loss_object=loss_object,
                                            clip_values=(0, 1), channels_first=False)
    params = {"eps_step": 1.0, "eps": 1.0}
    empirical_robustness = None
    if(tipo_ataque=='fgsm'):
        empirical_robustness = art.metrics.metrics.empirical_robustness(test_classifier, 
                                                                    data[:num_imgs], 
                                                                    str("fgsm")
                                                                   )
    elif(tipo_ataque=='hsj'):
        empirical_robustness = art.metrics.metrics.empirical_robustness(test_classifier, 
                                                                    data[:num_imgs], 
                                                                    str("hsj")
                                                                   )
        

    print(empirical_robustness)
    return empirical_robustness