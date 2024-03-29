import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from .LanguageCrossAttention import LangCrossAtt

from visualization_attention import visualization_attention


class Attention_ConTEXTual_Seg_Model(torch.nn.Module):
    def __init__(self, lang_model, n_channels, n_classes, bilinear=False):
        super(Attention_ConTEXTual_Seg_Model, self).__init__()

        self.lang_encoder = lang_model

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, bilinear)
        self.attention1 = Attention_block(512, 512, 256)
        self.lang_attn = LangCrossAtt(emb_dim=1024)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(512, bilinear)
        self.attention2 = Attention_block(256, 256, 128)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = Up(256, bilinear)
        self.attention3 = Attention_block(128, 128, 64)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(128, bilinear)
        self.attention4 = Attention_block(64, 64, 32)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_classes)

        self.lang_proj1 = nn.Linear(1024, 512)
        self.lang_attn1 = LangCrossAtt(emb_dim=512)
        self.lang_proj2 = nn.Linear(1024, 256)
        self.lang_attn2 = LangCrossAtt(emb_dim=256)
        self.lang_proj3 = nn.Linear(1024, 128)
        self.lang_attn3 = LangCrossAtt(emb_dim=128)
        self.lang_proj4 = nn.Linear(1024, 64)
        self.lang_attn4 = LangCrossAtt(emb_dim=64)

    def forward(self, img, ids, mask, token_type_ids, target_batch):
        # for roberta
        # lang_output = self.lang_encoder(ids, mask, token_type_ids)
        # word_rep = lang_output[0]
        # report_rep = lang_output[1]
        # lang_rep = word_rep

        # for t5
        encoder_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = encoder_output.last_hidden_state
        lang_rep = pooled_sentence

        x1 = self.inc(img)
        print(f"x1 size: {x1.size()}")
        x2 = self.down1(x1)
        print(f"x2 size: {x2.size()}")
        x3 = self.down2(x2)
        print(f"x3 size: {x3.size()}")
        x4 = self.down3(x3)
        print(f"x4 size: {x4.size()}")
        x5 = self.down4(x4)
        print(f"x5 size: {x5.size()}")


        decode1_before = self.up1(x5)
        print(f"decode1_before: {decode1_before.size()}")
        lang_rep1 = self.lang_proj1(lang_rep)
        print(f"lang_rep1 size: {lang_rep1.size()}")
        decode1, att_matrix1 = self.lang_attn1(lang_rep=lang_rep1, vision_rep=decode1_before)
        print(f"decode1_after: {decode1.size()}")
        # How is used to be done, swapping for testing
        x4 = self.attention1(decode1, x4)
        #print(f"x4 after vision attention: {x4.size()}")
        #print(f"decode1 size: {decode1.size()}")
        x = concatenate_layers(decode1, x4)
        #print(f"after first concatentation: {x.size()}")
        x = self.up_conv1(x)

        #print(f"before up2 is applied: {x.size()}")

        decode2_before = self.up2(x)
        #print(f"after up2 is applied: {decode2_before.size()}")
        lang_rep2 = self.lang_proj2(lang_rep)
        #print(f"lang_rep2 size: {lang_rep2.size()}")
        decode2, att_matrix2 = self.lang_attn2(lang_rep=lang_rep2, vision_rep=decode2_before)

        #print(f"output of lang_attn2: {decode2.size()}")

        x3 = self.attention2(decode2, x3)

        #print(f"output of vision attention 2: {x3.size()}")

        x = concatenate_layers(decode2, x3)
        #print(f"after second concatentation: {x.size()}")
        x = self.up_conv2(x)

        decode3_before = self.up3(x)

        #print(f"after third decoder double up sample: {decode3_before.size()}")

        lang_rep3 = self.lang_proj3(lang_rep)
        #print(f"lang_rep3 size: {lang_rep3.size()}")
        decode3, att_matrix3 = self.lang_attn3(lang_rep=lang_rep3, vision_rep=decode3_before)
        #print(f"output of lang_attn3: {decode3.size()}")

        x2 = self.attention3(decode3, x2)
        #print(f"output of vision attention 3: {x2.size()}")

        x = concatenate_layers(decode3, x2)

        #print(f"after third concatentation: {x.size()}")

        x = self.up_conv3(x)

        decode4_before = self.up4(x)

        #print(f"after fourth decoder double up sample: {decode4_before.size()}")
        lang_rep4 = self.lang_proj4(lang_rep)
        #print(f"lang_rep4 size: {lang_rep4.size()}")
        decode4, att_matrix4 = self.lang_attn4(lang_rep=lang_rep4, vision_rep=decode4_before)
        #print(f"output of lang_attn4: {decode4.size()}")

        x1 = self.attention4(decode4, x1)
        #print(f"output of vision attention 1: {x1.size()}")
        x = concatenate_layers(decode4, x1)
        #print(f"after fourth concatentation: {x.size()}")
        x = self.up_conv4(x)

        #print(f"after final upconv4: {x.size()}")

        logits = self.outc(x)

        visualization_attention(img, decode1_before, decode1, lang_rep1, att_matrix1, target_batch, logits, "1")
        visualization_attention(img, decode2_before, decode2, lang_rep2, att_matrix2, target_batch, logits, "2")
        visualization_attention(img, decode3_before, decode3, lang_rep3, att_matrix3, target_batch, logits, "3")
        visualization_attention(img, decode4_before, decode4, lang_rep4, att_matrix4, target_batch, logits, "4")


        print(f"logits shape: {logits.size()}")
        return logits


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.channelReduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.channelReduce(x1)  # remove this when I got back to non bilinear
        return x1


def concatenate_layers(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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
