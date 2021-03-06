import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import os
from collections import OrderedDict
from .LanguageCrossAttention import LangCrossAtt


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResAttNetUNet(nn.Module):
    def __init__(self, lang_model, n_class, dir_base): #lang_model
        super().__init__()

        self.lang_encoder = lang_model

        self.base_model = models.resnet50(pretrained=False)
        save_path = os.path.join(dir_base, 'Zach_Analysis/models/resnet/resnet50')
        pretrained_path = os.path.join(dir_base,
                                       'Zach_Analysis/models/candid_pretrained_models/roberta/candid_best_contrastive')
        pretrained_model = True
        if pretrained_model:
            state_dict = torch.load(pretrained_path)
            # seg_model.encoder.load_state_dict(ckpt)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[6:]  # remove `model.`
                new_state_dict[name] = v

            # delete extra layers
            del new_state_dict["_embedder.weight"]
            del new_state_dict["_embedder.bias"]
            del new_state_dict["embedder.weight"]

            #print(type(state_dict["global_embedder.weight"]))
            #new_state_dict["fc.bias"] = state_dict["global_embedder.bias"]
            #new_state_dict["fc.weight"] = state_dict["local_embedder.weight"]
            bias = torch.empty(1000)
            #load_bias = nn.init.xavier_uniform(bias)
            weight = torch.empty(1000, 2048)
            load_weight = nn.init.xavier_uniform_(weight)
            new_state_dict["fc.bias"] = bias
            new_state_dict["fc.weight"] = load_weight

            # load in the parameters
            self.base_model.load_state_dict(new_state_dict)
            #self.base_model.load_state_dict(torch.load(pretrained_path))
        else:
            self.base_model.load_state_dict(torch.load(save_path))


        self.base_layers = list(self.base_model.children())

        #print(self.base_layers)
        #print(len(self.base_layers))

        #self.lang_encoder = lang_model
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)


        # layers from other version
        bilinear = False

        self.up1 = Up(2048, bilinear)
        self.attention1 = Attention_block(1024, 1024, 512)
        # self.multiplicativeAttention = DotProductAttention(hidden_dim=10)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=1)

        lang_dimension = 1024
        self.lang_proj1 = nn.Linear(lang_dimension, 1024)
        self.lang_attn1 = LangCrossAtt(emb_dim=1024)
        self.up_conv1 = DoubleConv(2048, 1024)

        self.up2 = Up(1024, bilinear)
        self.attention2 = Attention_block(512, 512, 256)

        self.lang_proj2 = nn.Linear(lang_dimension, 512)
        self.lang_attn2 = LangCrossAtt(emb_dim=512)

        self.up_conv2 = DoubleConv(1024, 512)

        self.up3 = Up(512, bilinear)
        self.attention3 = Attention_block(256, 256, 128)

        self.lang_proj3 = nn.Linear(lang_dimension, 256)
        self.lang_attn3 = LangCrossAtt(emb_dim=256)

        self.up_conv3 = DoubleConv(512, 256)

        self.up4 = Up(256, bilinear)
        self.up4_1x1 = convrelu(128, 64, 1, 0)

        self.attention4 = Attention_block(64, 64, 64)

        self.lang_proj4 = nn.Linear(lang_dimension, 64)
        self.lang_attn4 = LangCrossAtt(emb_dim=64)

        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_class)

    def forward(self, input, ids, mask, token_type_ids):

        # language representations
        #for t5
        lang_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = lang_output.last_hidden_state
        #pooled_sentence = torch.mean(pooled_sentence, dim=1)
        lang_rep = pooled_sentence

        #for roberta
        #lang_output = self.lang_encoder(ids, mask, token_type_ids)
        #lang_rep = lang_output[1]

        # vision representations
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        #layer4 = self.lang_attn0(lang_rep=lang_rep, vision_rep=layer4)

        decode1 = self.up1(layer4)
        layer3 = self.attention1(decode1, layer3)

        # language attention
        # this one does not have a projection because it is already 1024
        lang_rep1 = self.lang_proj1(lang_rep)
        layer3 = self.lang_attn1(lang_rep=lang_rep1, vision_rep=layer3)

        x = concatenate_layers(decode1, layer3)
        x = self.up_conv1(x)

        decode2 = self.up2(x)
        layer2 = self.attention2(decode2, layer2)

        # language attention
        lang_rep2 = self.lang_proj2(lang_rep)
        layer2 = self.lang_attn2(lang_rep=lang_rep2, vision_rep=layer2)

        x = concatenate_layers(decode2, layer2)
        x = self.up_conv2(x)

        decode3 = self.up3(x)
        layer1 = self.attention3(decode3, layer1)

        # language attention
        lang_rep3 = self.lang_proj3(lang_rep)
        layer1 = self.lang_attn3(lang_rep=lang_rep3, vision_rep=layer1)

        x = concatenate_layers(decode3, layer1)
        x = self.up_conv3(x)

        decode4 = self.up4(x)
        decode4 = self.up4_1x1(decode4)
        layer0 = self.attention4(decode4, layer0)

        # language attention
        lang_rep4 = self.lang_proj4(lang_rep)
        layer0 = self.lang_attn4(lang_rep=lang_rep4, vision_rep=layer0)

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

"""
class LangCrossAtt(nn.Module):
    "add documentaiton"


    def __init__(self, emb_dim):
        super(LangCrossAtt, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1) #vdim=vdimension

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


        lang_rep = torch.unsqueeze(lang_rep, 1)
        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        #lang_rep = torch.swapaxes(lang_rep, 1, 2)

        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)

        # gets the attention weights and repeats them to have the same size as the total channels
        attn_output_weights = torch.swapaxes(attn_output_weights, 0, 1)
        #attn_output_weights = torch.swapaxes(attn_output_weights, 0, 2)
        attn_output_weights = attn_output_weights.repeat(1, 1, input_channel)

        # multiplies the attention to focus the vision rep based on the lang rep
        vision_rep = vision_rep * attn_output_weights
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)
        return out
        """
