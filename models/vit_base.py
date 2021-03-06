import os

import timm
import torch
import torch.nn as nn


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False, dir_base="/home/zmh001/r-fcb-isilon/research/Bradshaw/"):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch32_384", pretrained=False)
        self.classifier = nn.Linear(1000, n_classes)
        # self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        pretrained = True
        if pretrained:
            # MODEL_PATH = ("C:/Users/zmh001/Documents/vit_model/jx_vit_base_p16_224-80ecf9dd.pth/jx_vit_base_p16_224-80ecf9dd.pth")
            # MODEL_PATH = ('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/vit_model/jx_vit_base_p16_224-80ecf9dd.pth/jx_vit_base_p16_224-80ecf9dd.pth')
            # model_path = os.path.join(dir_base, 'Zach_Analysis/vit_model/jx_vit_base_p16_224-80ecf9dd.pth/jx_vit_base_p16_224-80ecf9dd.pth')
            model_path = os.path.join(dir_base, 'Zach_Analysis/vit_model/jx_vit_base_p32_384-830016f5.pth')
            self.model.load_state_dict(torch.load(model_path))
            print("is using the weights stored at this location")
        else:
            print("doesn't use saved weights, using random weights in vision")
        # self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # self.model.head = nn.Linear(self.model.head.in_features, 512)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
