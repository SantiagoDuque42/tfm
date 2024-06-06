import numpy as np

def cargar_train_test_data(path_ataques='adv_attacks/'):
    shuffled_train_data = np.load(path_ataques + 'shuffled_train_data.npy')
    shuffled_train_labels = np.load(path_ataques + 'shuffled_train_labels.npy')
    shuffled_test_data = np.load(path_ataques + 'shuffled_test_data.npy')
    shuffled_test_labels = np.load(path_ataques + 'shuffled_test_labels.npy')
    
    return shuffled_train_data, shuffled_train_labels, shuffled_test_data, shuffled_test_labels


def cargar_adv_data_cw(path_ataques='adv_attacks/'):
    adv_data_cw = np.load(path_ataques + 'test_adv_data_cw.npy')
    
    return adv_data_cw

def cargar_adv_data_fgsm(path_ataques='adv_attacks/'):
    data_adv_fgsm_01 = np.load(path_ataques + 'data_adv_fgsm_01.npy')
    data_adv_fgsm_02 = np.load(path_ataques + 'data_adv_fgsm_02.npy')
    data_adv_fgsm_03 = np.load(path_ataques + 'data_adv_fgsm_03.npy')
    data_adv_fgsm_005 = np.load(path_ataques + 'data_adv_fgsm_005.npy')
    data_adv_fgsm_0005 = np.load(path_ataques + 'data_adv_fgsm_0005.npy')
    
    return data_adv_fgsm_03, data_adv_fgsm_02, data_adv_fgsm_01, data_adv_fgsm_005, data_adv_fgsm_0005

def cargar_adv_data_deepfool(path_ataques='adv_attacks/'):
    adv_data_deepfool = np.load(path_ataques + 'adv_data_deepfool.npy')
    
    return adv_data_deepfool

def cargar_adv_data_hsj(path_ataques='adv_attacks/'):
    adv_data_hsj_combined = np.load(path_ataques + 'adv_data_hsj_combined.npy')
    
    return adv_data_hsj_combined


def cargar_shadow_data(path_ataques='adv_attacks/'):
    shadow_data_adv_fgsm = np.load(path_ataques + 'shadow_data_adv_fgsm.npy')
    shadow_data_adv_cw = np.load(path_ataques + 'shadow_data_adv_cw.npy')
    
    return shadow_data_adv_fgsm, shadow_data_adv_cw