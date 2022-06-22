import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import os

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResAttNetUNet(nn.Module):
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


        self.attention1 = Attention_block(1024, 1024, 512)
        self.attention2 = Attention_block(512, 512, 256)
        self.attention3 = Attention_block(256, 256, 64)

        self.up1 = Up(2048, bilinear=False)
        self.up2 = Up(1024, bilinear=False)
        self.up3 = Up(512, bilinear=False)
        self.up4 = Up(256, bilinear=False)
        self.up5 = Up(128, bilinear=False)





        bilinear = False
        # layers from other versino
        self.up1 = Up(2048, bilinear)
        #self.attention1 = Attention_block(512, 512, 256)
        # self.multiplicativeAttention = DotProductAttention(hidden_dim=10)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=1)
        self.lang_attn = LangCrossAtt(emb_dim=1024)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(1024, bilinear)
        #self.attention2 = Attention_block(256, 256, 128)
        self.up_conv2 = DoubleConv(512, 256)

        #self.up3 = Up(512, bilinear)
        self.attention3 = Attention_block(128, 128, 64)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(256, bilinear)
        self.attention4 = Attention_block(64, 64, 32)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_class)







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

        #x5 = self.lang_attn(lang_rep=lang_rep, vision_rep=x5)

        decode1 = self.up1(layer4)

        #print("x5 shape")
        #print(x5.size())
        #x4 = self.attention1(decode1, x4)
        #test = self.multiplicativeAttention(lang_output[1], decode1)
        #x5 = torch.swapaxes(x5, 0, 1)

        #x5 = torch.flatten(x5, start_dim=2)
        #x5 = torch.swapaxes(x5, 2, 0)


        #lang_rep = torch.swapaxes(lang_rep, 0, 1)
        #lang_rep = torch.swapaxes(lang_rep, 1, 2)

        #print("lang_rep")
        #print(lang_rep.size())

        #test_att, test_other = self.multihead_attn(query = decode1, key = lang_rep, value = lang_rep)

        #test = self.multihead_attn(query=lang_rep, key=decode1, value=decode1)
        #print("attention size")
        #print(test_att.size())
        #print(test_other.size())
        x = concatenate_layers(decode1, layer3)
        x = self.up_conv1(x)

        decode2 = self.up2(x)
        #x3 = self.attention2(decode2, x3)
        x = concatenate_layers(decode2, layer2)
        x = self.up_conv2(x)

        decode3 = self.up3(x)
        #x2 = self.attention3(decode3, x2)
        x = concatenate_layers(decode3, layer1)
        x = self.up_conv3(x)

        decode4 = self.up4(x)
        #x1 = self.attention4(decode4, x1)
        x = concatenate_layers(decode4, layer0)
        x = self.up_conv4(x)

        logits = self.outc(x)

        return logits


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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def concatenate_layers(x1, x2):

    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x

class LangCrossAtt(nn.Module):
    "add documentaiton"


    def __init__(self, emb_dim):
        super(LangCrossAtt, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1)

    def forward(self, lang_rep, vision_rep):

        # gets all of the dimensions to be used in the attention
        input_batch = vision_rep.size()[0]
        input_channel =  vision_rep.size()[1]
        input_width = vision_rep.size()[2]
        input_height =  vision_rep.size()[3]

        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)

        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        lang_rep = torch.swapaxes(lang_rep, 1, 2)

        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)

        # gets the attention weights and repeats them to have the same size as the total channels
        attn_output_weights = torch.swapaxes(attn_output_weights, 0, 1)
        attn_output_weights = attn_output_weights.repeat(1, 1, input_channel)

        # multiplies the attention to focus the vision rep based on the lang rep
        vision_rep = vision_rep * attn_output_weights
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)
        return out
