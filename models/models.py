# import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import models

class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""

    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel must implement forward.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers (list): List of names for the boundary layers.
            lrs (list): List of learning rates for each parameter group, from earlier to later layers.

        Returns:
            list: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError('Got {} param groups, but {} learning rates'.format(boundary_layers + 1, len(lrs)))

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i+2]
            param_groups.append({'params': gen_params(start, end), 'lr': lrs[i]})

        return param_groups

    
class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """
    def __init__(self, model_fn, task_sequence, model_args):
        super(TorchVisionModel, self).__init__()

        self.task_sequence = task_sequence
        self.model_uncertainty = model_args.model_uncertainty
        self.frontal_lateral = model_args.frontal_lateral
        self.upweight_tb = model_args.upweight_tb

        self.model = model_fn(pretrained=model_args.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

        len_task_sequence = len(task_sequence)
        
        if self.model_uncertainty:
            num_outputs = 3 * len_task_sequence
        elif self.frontal_lateral:
            num_outputs = 1
        else:
            num_outputs = len_task_sequence

        self.num_outputs = len_task_sequence
        
    def forward(self, x_list):
        if len(x_list[1][0]) > 0:
            x = self.model.features(x_list[0])
            x = F.relu(x, inplace=True)
            x = self.pool(x).view(x.size(0), -1)

            # The first classifier, that predicts the occurence of certain clinical findings
            x_findings = self.model.classifier_f(x)

            # The second classifier, that predicts TB based on image and covars
            x_tb = F.relu(self.model.reduce_img_ftrs(x))

            if len(x_list[1][0]) > 0:
                x_list[1] = x_list[1].float()
                x_tb = F.relu(self.model.first_covar_img_layer(
                    torch.cat((x_tb, x_list[1]), 1)))
                x_tb = self.model.classifier(x_tb)
            else:
                x_tb = F.relu(self.model.first_covar_img_layer(x_tb))
                x_tb = self.model.classifier(x_tb)

            x = torch.cat((x_tb, x_findings), 1)
            return x
        else:
            x = self.model.features(x_list[0])
            x = F.relu(x, inplace=True)
            x = self.pool(x).view(x.size(0), -1)
            if len(x_list[1][0]) > 0:
                x_list[1] = x_list[1].float()
                x = self.model.classifier(torch.cat((x, x_list[1]), 1))
            else:
                x = self.model.classifier(x)
            return x

    def transform_model_shape(self, len_task_sequence, num_covars=0):

        if num_covars > 0:
            num_ftrs = self.model.classifier.in_features + num_covars

            if self.model_uncertainty:
                num_outputs = 3 * len_task_sequence
            elif self.frontal_lateral:
                num_outputs = 1
            else:
                num_outputs = len_task_sequence

            num_combined_ftrs = 20 + num_covars
                
            self.model.classifier_f = nn.Linear(self.model.classifier.in_features,
                                                num_outputs - 1)
            self.model.reduce_img_ftrs = nn.Linear(self.model.classifier.in_features, 20)
            self.model.first_covar_img_layer = nn.Linear(num_combined_ftrs, 10)
            self.model.classifier = nn.Linear(10, 1)
        
        else:
            num_ftrs = self.model.classifier.in_features + num_covars
            if self.model_uncertainty:
                num_outputs = 3 * len_task_sequence
            elif self.frontal_lateral:
                num_outputs = 1
            else:
                num_outputs = len_task_sequence
            self.model.classifier = nn.Linear(num_ftrs, num_outputs)

        
class DenseNet121(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(DenseNet121, self).__init__(models.densenet121, task_sequence, model_args)


class DenseNet161(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(DenseNet161, self).__init__(models.densenet161, task_sequence, model_args)


class DenseNet201(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(DenseNet201, self).__init__(models.densenet201, task_sequence, model_args)


class ResNet101(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(ResNet101, self).__init__(models.resnet101, task_sequence, model_args)


class ResNet152(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(ResNet152, self).__init__(models.resnet152, task_sequence, model_args)


class Inceptionv3(TorchVisionModel):
    def __init__(self, task_sequence, model_args):
        super(Inceptionv3, self).__init__(models.densenet121, task_sequence, model_args)


