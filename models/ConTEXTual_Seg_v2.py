import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from .LanguageCrossAttention import LangCrossAtt


# from visualization_attention import visualization_attention


class Attention_ConTEXTual_Seg_Model_swap(torch.nn.Module):
    def __init__(self, lang_model, n_channels, n_classes, bilinear=False):
        super(Attention_ConTEXTual_Seg_Model_swap, self).__init__()

        self.lang_encoder = lang_model

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

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

        self.lang_proj1d = nn.Linear(1024, 512)
        self.lang_attn1d = LangCrossAtt(emb_dim=512)
        self.lang_proj2d = nn.Linear(1024, 256)
        self.lang_attn2d = LangCrossAtt(emb_dim=256)
        self.lang_proj3d = nn.Linear(1024, 128)
        self.lang_attn3d = LangCrossAtt(emb_dim=128)
        self.lang_proj4d = nn.Linear(1024, 64)
        self.lang_attn4d = LangCrossAtt(emb_dim=64)

        self.lang_proj1x = nn.Linear(1024, 512)
        self.lang_attn1x = LangCrossAtt(emb_dim=512)
        self.lang_proj2x = nn.Linear(1024, 256)
        self.lang_attn2x = LangCrossAtt(emb_dim=256)
        self.lang_proj3x = nn.Linear(1024, 128)
        self.lang_attn3x = LangCrossAtt(emb_dim=128)
        self.lang_proj4x = nn.Linear(1024, 64)
        self.lang_attn4x = LangCrossAtt(emb_dim=64)

    def forward(self, img, ids, mask, token_type_ids):
        # for roberta
        #lang_output = self.lang_encoder(ids, mask, token_type_ids)
        #word_rep = lang_output[0]
        #report_rep = lang_output[1]
        #lang_rep = word_rep

        # for t5
        encoder_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = encoder_output.last_hidden_state
        lang_rep = pooled_sentence

        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        decode1 = self.up1(x5)

        lang_rep1 = self.lang_proj1d(lang_rep)
        decode1 = self.lang_attn1d(lang_rep=lang_rep1, vision_rep=decode1)

        lang_rep1x = self.lang_proj1x(lang_rep)
        x4 = self.lang_attn1x(lang_rep=lang_rep1x, vision_rep=x4)

        # How is used to be done, swapping for testing
        x4 = self.attention1(decode1, x4)

        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        decode2 = self.up2(x)
        lang_rep2 = self.lang_proj2d(lang_rep)
        decode2 = self.lang_attn2d(lang_rep=lang_rep2, vision_rep=decode2)

        lang_rep2x = self.lang_proj2x(lang_rep)
        x3 = self.lang_attn2x(lang_rep=lang_rep2x, vision_rep=x3)

        x3 = self.attention2(decode2, x3)
        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        decode3 = self.up3(x)
        lang_rep3 = self.lang_proj3d(lang_rep)
        decode3 = self.lang_attn3d(lang_rep=lang_rep3, vision_rep=decode3)

        lang_rep3x = self.lang_proj3x(lang_rep)
        x2 = self.lang_attn3x(lang_rep=lang_rep3x, vision_rep=x2)

        x2 = self.attention3(decode3, x2)
        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        decode4 = self.up4(x)
        lang_rep4 = self.lang_proj4d(lang_rep)
        decode4 = self.lang_attn4d(lang_rep=lang_rep4, vision_rep=decode4)

        lang_rep4x = self.lang_proj4x(lang_rep)
        x1 = self.lang_attn4x(lang_rep=lang_rep4x, vision_rep=x1)

        x1 = self.attention4(decode4, x1)
        x = concatenate_layers(decode4, x1)
        x = self.up_conv4(x)

        logits = self.outc(x)

        # visualization_attention(img, decode4, lang_rep4, att_matrix, target_batch, logits)

        return logits


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


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        print("query:")
        print(query.size())
        print("value")
        print(value.size())
        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


"""
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

"""
