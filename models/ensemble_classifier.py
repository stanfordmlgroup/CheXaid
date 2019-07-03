import numpy as np
import torch


class EnsembleClassifier(object):
    """Class with nn.Module Class methods that is an ensemble of 
    nn.Module objects."""
    def __init__(self, models):
        self.models = models

    def to(self, device):
        for model in self.models:
            model = model.to(device)
        return self

    def eval(self):
        for model in self.models:
            model.eval()

    def forward(self, x):
        """Computes logits for each model.
        Args:
            x: Input to the forward pass.
        Return:
            list of logits, once for each model.
        """
        ensemble_logits = [model.forward(x)
                           for model in self.models]

        return ensemble_logits

    def sigmoid(self, ensemble_logits):
        """Applies sigmoid to logit output from each model in the ensemble,
        then averages them."""
        ensemble_probs = torch.stack([torch.sigmoid(ensemble_logit)
                                      for ensemble_logit in ensemble_logits])

        return torch.mean(ensemble_probs, 0)