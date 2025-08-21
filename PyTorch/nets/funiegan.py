"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.attention import SpatioChannelAttention 

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        if x.shape[2:] != skip_input.shape[2:]:
         h = min(x.shape[2], skip_input.shape[2])
         w = min(x.shape[3], skip_input.shape[3])
         x = x[:, :, :h, :w]
         skip_input = skip_input[:, :, :h, :w]
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        
        # at multiple stages in the network to enrich the feature representations"
        self.sca_128 = SpatioChannelAttention(128)  # After down2
        self.sca_256_1 = SpatioChannelAttention(256)  # After down3
        self.sca_256_2 = SpatioChannelAttention(256)  # After down4
        self.sca_256_3 = SpatioChannelAttention(256)  # After down5 (bottleneck)
        
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d2_attention = self.sca_128(d2)
        
        d3 = self.down3(d2_attention)
        d3_attention = self.sca_256_1(d3)
        
        d4 = self.down4(d3_attention)
        d4_attention = self.sca_256_2(d4)
        
        d5 = self.down5(d4_attention)
        d5_attention = self.sca_256_3(d5)
        
        u1 = self.up1(d5_attention, d4_attention)
        u2 = self.up2(u1, d3_attention)
        u3 = self.up3(u2, d2_attention)
        u45 = self.up4(u3, d1)
        return self.final(u45)
        
       
class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        if img_A.shape[2:] != img_B.shape[2:]:
         h = min(img_A.shape[2], img_B.shape[2])
         w = min(img_A.shape[3], img_B.shape[3])
         img_A = img_A[:, :, :h, :w]
         img_B = img_B[:, :, :h, :w]
         
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

