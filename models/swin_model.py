import torch
import torch.nn as nn
import timm

class SwinModel(torch.nn.Module):
    def __init__(self, backbone):
        print("hi")
        super(SwinModel, self).__init__()

        self.backbone = backbone

    def forward(self, img):

        output = self.backbone(img)
        print(output)

        return output

