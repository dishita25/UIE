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
        print(f"d1 shape: {d1.shape}")
        d2 = self.down2(d1)
        print(f"d2 shape: {d2.shape}")
        d2_attention = self.sca_128(d2)
        print(f"d2_attention shape: {d2_attention.shape}")
        
        d3 = self.down3(d2_attention)
        print(f"d3 shape: {d3.shape}")
        d3_attention = self.sca_256_1(d3)
        print(f"d3_attention shape: {d3_attention.shape}")
        
        d4 = self.down4(d3_attention)
        print(f"d4 shape: {d4.shape}")
        d4_attention = self.sca_256_2(d4)
        print(f"d4_attention shape: {d4_attention.shape}")
        
        d5 = self.down5(d4_attention)
        print(f"d4 shape: {d4.shape}")
        d5_attention = self.sca_256_3(d5)
        print(f"d5_attention shape: {d5_attention.shape}")
        
        u1 = self.up1(d5_attention, d4_attention)
        print(f"u1 shape: {u1.shape}")
        u2 = self.up2(u1, d3_attention)
        print(f"u2 shape: {u2.shape}")
        u3 = self.up3(u2, d2_attention)
        print(f"u3 shape: {u3.shape}")
        u45 = self.up4(u3, d1)
        print(f"u45 shape: {u45.shape}")
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
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

