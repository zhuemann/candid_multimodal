import torch
import torch.nn as nn
from torchvision import models
import os

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, dir_base): #lang_model
        super().__init__()

        self.base_model = models.resnet50(pretrained=False)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/resnet/resnet50')
        self.base_model.load_state_dict(torch.load(save_path))


        self.base_layers = list(self.base_model.children())

        #print(self.base_layers)
        #print(len(self.base_layers))

        #self.lang_encoder = lang_model
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0) # was (64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0) # was (64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0) # was (128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0) #was (256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0) #was (512, 512, 1, 0)
        #self.layer5 = self.base_layers[8]  # size=(N, 1024, x.H/64, x.W/64)

        #self.layer5_1x1 = convrelu(1024, 1024, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        #numbers for resnet34
        #self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        #self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        #self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        #self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        #self.conv_original_size0 = convrelu(3, 64, 3, 1)
        #self.conv_original_size1 = convrelu(256, 256, 3, 1)
        #self.conv_original_size2 = convrelu(256 + 512, 64, 3, 1)
        #numbers for resnet34
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)


        #self.attention1 = Attention_block(1024, 1024, 512)
        #self.attention2 = Attention_block(512, 512, 256)
        #self.attention3 = Attention_block(256, 256, 64)

        #self.up1 = Up(2048, bilinear=False)
        #self.up2 = Up(1024, bilinear=False)
        #self.up3 = Up(512, bilinear=False)
        #self.up4 = Up(256, bilinear=False)
        #self.up5 = Up(128, bilinear=False)





    def forward(self, input, ids, mask, token_type_ids):

        #lang_output = self.lang_encoder(ids, mask, token_type_ids)
        #lang_rep = torch.unsqueeze(torch.unsqueeze(lang_output[1], 2), 3)
        #lang_rep = lang_rep.repeat(1, 2, 8, 8)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        print(f"layer0: {layer0.size()}")
        print(f"layer1: {layer1.size()}")
        print(f"layer2: {layer2.size()}")
        print(f"layer3: {layer3.size()}")
        print(f"layer4: {layer4.size()}")

        layer4 = self.layer4_1x1(layer4)
        print(f"layer4 size after 1x1 {layer4.size()}")

        #print(f"after upsampling {layer4.size()}")
        x = self.upsample(layer4)
        print(f"x value after upsampling layer4 {x.size()}")

        test = self.up1(layer4)
        print(f"test dimensions: {test.size()}")

        layer3 = self.layer3_1x1(layer3)
        #print(f"layer3 after 1x1 {layer3.size()}")

        #attention goes here

        #print("layer 4 after 1x1")
        #print(f"layer 4 size into attention{layer4.size()}")
        #print(layer3.size())
        #print(f"layer 3 size into attention{layer3.size()}")
        #print(f"x size into attention{x.size()}")

        #layer3 = self.attention1(layer3, x)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        #attention goes here
        #layer2 = self.attention2(layer2, x)

        x = torch.cat([x, layer2], dim=1)
        #print(f"conv_up2 shape: {x.size()}")
        x = self.conv_up2(x)
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)

        #layer1 = self.attention1(layer1, x)

        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)

        #layer0 = self.attention3(layer0, x)


        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class Attention_block(nn.Module):

    """https://github.com/LeeJunHyun/Image_Segmentation"""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.x_layer = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        gated_conv = self.gate(g)
        layer_conv = self.x_layer(x)
        psi = self.relu(gated_conv + layer_conv)
        psi = self.psi(psi)

        return x * psi


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)

        return x1
