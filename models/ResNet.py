import os

import torch.nn as nn
from torchvision import models as models_2d
import torch


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


def resnet_34(pretrained=True):
    model = models_2d.resnet34(pretrained=False)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True, dir_base="/Users/kritigoyal/Documents/CS_769_NLP/"): #"/Users/kritigoyal/Documents/CS_769_NLP/"
    if pretrained:
        save_path = os.path.join(dir_base, 'models/resnet/resnet50')
        model = models_2d.resnet50(pretrained=False)
        model.load_state_dict(torch.load(save_path))
    else:
        model = models_2d.resnet50(pretrained=False)

    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024
