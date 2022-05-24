import torch
import torch.nn as nn
import torch.nn.functional as F



class ConTEXTual_seg_model(torch.nn.Module):
    def __init__(self, lang_model, n_channels, n_classes, bilinear=False):

        super(ConTEXTual_seg_model, self).__init__()

        self.lang_encoder = lang_model

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)

        self.flatten = nn.Flatten(1,-1)
        self.linear = nn.Linear(64*16*16+1024, 64*16*16)
        #self.combine = OutConv(2048, 1024)
        # self.up0 = Up(1024, 1024 // factor, bilinear)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
        self.outc = OutConv(4, n_classes)

        """
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #self.down4 = Down(32, 64)
        #self.down5 = Down(64, 128)
        #self.down6 = Down(128, 256)
        #self.down7 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.combine = OutConv(2048, 1024)
        #self.up0 = Up(1024, 1024 // factor, bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.up5 = Up(64, 32, bilinear)
        #self.up6 = Up(32, 16, bilinear)
        #self.up7 = Up(16, 8, bilinear)
        #self.up8 = Up(8, 4, bilinear)
        #self.outc = OutConv(64, n_classes)
        self.outc = OutConv(64, n_classes)
        """

    def forward(self, img, ids, mask, token_type_ids):

        lang_output = self.lang_encoder(ids, mask, token_type_ids)
        lang_rep = torch.unsqueeze(torch.unsqueeze(lang_output[1], 2), 3)
        lang_rep = lang_rep.repeat(1, 1, 16, 16)
        #print(lang_rep.size())
        #size = lang_rep.size()

        #zeros = torch.zeros(size, device=torch.device('cuda:0') )
        """
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        joint_rep = torch.cat((x9, lang_rep), dim=1)

        x_comb = self.combine(joint_rep)

        x = self.up1(x_comb, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        logits = self.outc(x)
        """

        #print("forwards")
        #print(img.size())
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        joint_rep = torch.cat((x5, lang_rep), dim=1)
        #joint_rep = torch.cat((x5, zeros), dim=1)


        joint_rep = self.flatten(joint_rep)
        print(joint_rep.size())
        x5 = self.linear(joint_rep)
        #x5 = self.combine(joint_rep)
        #print(x5.size())
        #print(lang_rep.size())

        #x = self.up0(x5, lang_rep)
        #x = self.up1(x, x4)

        #x = self.up1(zeros, x4)
        #x = self.up1(lang_rep, x4)
        x = self.up1(x5, x4)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)


        return logits



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

"""
class DownSample1x1(nn.Module):

    def __init__(selfself, in_channels, out_channels):
        super().__init__()

        self.onexone = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding = 0)

    def forward(self, x):
        return self.onexone(x)
"""


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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)