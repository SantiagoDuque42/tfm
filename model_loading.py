import numpy as np
from tensorflow import keras
import defensas.modelo_detector
import defensas.modelo_destilado
import defensas.modelo_ensemble_destilado
import data_loading
import utils

def cargar_baseline_model(path_to_models_folder="modelos/"):
    baseline_model = keras.models.load_model(path_to_models_folder+'modelo_base_1.keras')
    return baseline_model

def cargar_pgd_model(path_to_models_folder="modelos/"):
    pgd_model = keras.models.load_model(path_to_models_folder+'modelo_pgd.keras')
    return pgd_model

def cargar_naive_model(path_to_models_folder="modelos/"):
    naive_model = keras.models.load_model(path_to_models_folder+'naive_model.keras')
    return naive_model

def cargar_naive_2_model(path_to_models_folder="modelos/"):
    naive_model = keras.models.load_model(path_to_models_folder+'naive_model_2_order.keras')
    return naive_model

def cargar_ensemble_model(path_to_models_folder="modelos/"):
    ensemble_model = keras.models.load_model(path_to_models_folder+'modelo_ensemble_fgsm.keras')   
    return ensemble_model

def cargar_distilled_ensemble_model(path_to_models_folder="modelos/"):
    destilador = defensas.modelo_ensemble_destilado.Modelo_Ensemble_Destilado()
    train_data, train_labels, test_data, test_labels = data_loading.cargar_train_test_data()
    
    destilador_combinado = destilador.get_modelo(train_data, train_labels)
    destilador_combinado.load_weights(path_to_models_folder+"destilador_combinado_1")
    
    return destilador_combinado
    
def cargar_detector_model(path_to_models_folder="modelos/"):
    detector = defensas.modelo_detector.Modelo_Detector()
    modelo_detector = detector.get_modelo()
    
    modelo_detector.load_weights(path_to_models_folder+"modelo_detector_combinado_pesos")
    return modelo_detector
    
def cargar_distilled_model(path_to_models_folder="modelos/"):
    destilado = defensas.modelo_destilado.Modelo_Destilado()
    modelo_destilado = destilado.get_modelo()
    modelo_destilado.load_weights(path_to_models_folder +"destilador_base_pesos")
    return modelo_destilado
    
