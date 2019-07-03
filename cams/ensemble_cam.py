from .grad_cam import GradCAM

import torch
import numpy as np

from models import EnsembleClassifier

class EnsembleCAM(object):
    """Class for generating CAMs using an ensemble."""
    def __init__(self, model, device):

        super(EnsembleCAM, self).__init__()

        self.device = device
        self.model = model
        if isinstance(self.model, EnsembleClassifier):
            self.loaded_model_iterator = self.model.models
        else:
            self.loaded_model_iterator = self.model.loaded_model_iterator(task)
        self.grad_cams = [GradCAM(loaded_model, self.device) for loaded_model in self.loaded_model_iterator]

    def get_cam(self, x, task_id, task, covars=None):

        ensemble_probs = []
        cams = []
        
        for grad_cam in self.grad_cams:
            probs = grad_cam.forward(x, covars)
            grad_cam.backward(idx=task_id)
            cam = grad_cam.extract_cam()[0]
            ensemble_probs.append(probs)
            cams.append(cam)
            
        probs = np.mean(ensemble_probs, axis=0)
        sorted_probs = np.sort(probs, axis=0)[::-1]
        idx = np.argsort(probs, axis=0)[::-1]
        cam = np.mean(cams, axis=0)

        return sorted_probs, idx, cam
