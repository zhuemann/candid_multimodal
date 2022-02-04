from torchvision import models
import torch
import segmentation_models_pytorch as smp
import os

class VGG16():
    def __init__(self, n_classes, pretrained=False, dir_base="/home/zmh001/r-fcb-isilon/research/Bradshaw/"):
        super(VGG16, self).__init__()

        self.model = models.vgg16_bn()
        model_path = os.path.join(dir_base, 'Zach_Analysis/models/vgg16/vgg16_bn.pth')
        self.model.load_state_dict(torch.load(model_path))
        #self.classifier = torch.nn.Linear(1000, n_classes)

        #num_features = self.model.classifier[6].in_features
        #features = list(VGG16.classifier.children())[:-1]  # Remove last layer
        #features.extend([torch.nn.Linear(num_features, 2)])  # Add our layer with 4 outputs
        #VGG16.classifier = torch.nn.Sequential(*features)

    def forward(self, x):
        x = self.model(x)

        x = self.classifier(x)
        return x