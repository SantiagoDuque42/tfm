from art.attacks.evasion import FastGradientMethod, FrameSaliencyAttack, CarliniLInfMethod, HopSkipJump
from art.utils import to_categorical
from art.estimators.classification import TensorFlowV2Classifier
import model_loading
import utils


class Ataque_fgsm():
        
    def __init__(self, model):
        
        model = model
        loss_object, optimizer = utils.get_model_objects()           
        self.classifier = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(50,50,3), loss_object=loss_object,
                                    clip_values=(0, 1), channels_first=False)   
        
        
        
    def generar_ataque_fgsm(self, data, epsilon=0.3):
        self.attack = FastGradientMethod(estimator=self.classifier, eps=epsilon)
        adv_fgsm = self.attack.generate(data)
        return adv_fgsm
                                        

class Ataque_cw():
          
    def __init__(self, model):
        
        model = model
        loss_object, optimizer = utils.get_model_objects()
        self.classifier = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(50,50,3), loss_object=loss_object,
                                    clip_values=(0, 1), channels_first=False)   
        
        
    def generar_ataque_cw(self, data, max_iter=5, learning_rate=0.01):
        self.attack = CarliniLInfMethod(classifier=self.classifier,
                              max_iter=max_iter,
                              learning_rate=learning_rate,
                              initial_const=1e0,
                              largest_const=15e-1)
        
        adv_cw = self.attack.generate(data)
        return adv_cw
