"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.HVI_transform import RGB_HVI
from nets.LCA import *



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
            # nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)
        self.up = nn.Conv2d(out_size*2,out_size,kernel_size=1,stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.up(x)
        x = self.relu(x)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3, channels=[36, 36, 72, 144, 288], heads=[1, 2, 4, 8, 16]):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
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
        
        # *********Altered Architecture************
        
        [ch1, ch2, ch3, ch4, ch5] = channels
        [head1, head2, head3, head4, head5] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(        # 3 to 36 channels, H and W remain same
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        
        self.HVE_block1 = UNetDown(ch1, ch2)    # 36 to 36 channels, H/2 and W/2
        self.HVE_block2 = UNetDown(ch2, ch3)    # 36 to 72 channels, H/4 and W/4
        self.HVE_block3 = UNetDown(ch3, ch4)    # 72 to 144 channels, H/8 and W/8
        self.HVE_block4 = UNetDown(ch4, ch5)    # 144 to 288 channels, H/16 and W/16
        
        self.HVD_block4 = UNetUp(ch5, ch4)      # 288 to 144 channels, H/8 and W/8
        self.HVD_block3 = UNetUp(ch4, ch3)      # 288 to 72 channels, H/4 and W/4
        self.HVD_block2 = UNetUp(ch3, ch2)      # 72 to 36 channels, H/2 and W/2
        self.HVD_block1 = UNetUp(ch2, ch1)      # 36 to 36 channels, H and W
        
        self.HVD_block0 = nn.Sequential(        # 36 to 2 channels, H and W remain same
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(         # 1 to 36 channels, H and W remain same
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        
        self.IE_block1 = UNetDown(ch1, ch2)    # 36 to 36 channels, H/2 and W/2
        self.IE_block2 = UNetDown(ch2, ch3)    # 36 to 72 channels, H/4 and W/4
        self.IE_block3 = UNetDown(ch3, ch4)    # 72 to 144 channels, H/8 and W/8
        self.IE_block4 = UNetDown(ch4, ch5)    # 144 to 288 channels, H/16 and W/16
        
        self.ID_block4 = UNetUp(ch5, ch4)      # 288 to 144 channels, H/8 and W/8
        self.ID_block3 = UNetUp(ch4, ch3)      # 288 to 72 channels, H/4 and W/4
        self.ID_block2 = UNetUp(ch3, ch2)      # 72 to 36 channels, H/2 and W/2
        self.ID_block1 = UNetUp(ch2, ch1)      # 36 to 36 channels, H and W
        
        self.ID_block0 =  nn.Sequential(       # 36 to 1 channel, H and W remain the same 
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        # Encoder LCA - HV
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch5, head5)
        # Decoder LCA - HV
        self.HV_LCA5 = HV_LCA(ch5, head5)
        self.HV_LCA6 = HV_LCA(ch4, head4)
        self.HV_LCA7 = HV_LCA(ch3, head3)
        self.HV_LCA8 = HV_LCA(ch2, head2)
        
        # Encoder LCA - I
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch5, head5)
        # Decoder LCA - I
        self.I_LCA5 = I_LCA(ch5, head5)
        self.I_LCA6 = I_LCA(ch4, head4)
        self.I_LCA7 = I_LCA(ch3, head3)
        self.I_LCA8 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()  # To convert from RGB to HVI

    def forward(self, x):
        # hvi = self.trans.HVIT(x)
        # d1 = self.down1(hvi)
        # d2 = self.down2(d1)
        # d3 = self.down3(d2)
        # d4 = self.down4(d3)
        # d5 = self.down5(d4)
        # u1 = self.up1(d5, d4)
        # u2 = self.up2(u1, d3)
        # u3 = self.up3(u2, d2)
        # u45 = self.up4(u3, d1)
        # output_hvi = self.final(u45) + hvi
        # output_rgb = self.trans.PHVIT(output_hvi)
        
        # return output_rgb
        
        # **********New Forward***************
        print("Original RGB image")
        print(x)
        hvi = self.trans.HVIT(x)        
        i = hvi[:, 2, :, :].unsqueeze(1)
        print("HVI image after transform")
        print(hvi)
        
        # Level 0: Initial processing
        hv_enc0 = self.HVE_block0(hvi)          # HV: (batch, 3, H, W) -> (batch, 36, H, W)
        i_enc0 = self.IE_block0(i)              # I: (batch, 1, H, W) -> (batch, 36, H, W)
        
        # Level 1: First downsampling
        hv_enc1 = self.HVE_block1(hv_enc0)      # HV: (batch, 36, H/2, W/2)
        i_enc1 = self.IE_block1(i_enc0)         # I: (batch, 36, H/2, W/2)
        
        # Cross-attention at level 1
        hv_enc1 = self.HV_LCA1(hv_enc1, i_enc1)
        i_enc1 = self.I_LCA1(i_enc1, hv_enc1)
        
        # Level 2: Second downsampling
        hv_enc2 = self.HVE_block2(hv_enc1)      # HV: (batch, 72, H/4, W/4)
        i_enc2 = self.IE_block2(i_enc1)         # I: (batch, 72, H/4, W/4)
        
        # Cross-attention at level 2
        hv_enc2 = self.HV_LCA2(hv_enc2, i_enc2)
        i_enc2 = self.I_LCA2(i_enc2, hv_enc2)
        
        # Level 3: Third downsampling
        hv_enc3 = self.HVE_block3(hv_enc2)      # HV: (batch, 144, H/8, W/8)
        i_enc3 = self.IE_block3(i_enc2)         # I: (batch, 144, H/8, W/8)
        
        # Cross-attention at level 3
        hv_enc3 = self.HV_LCA3(hv_enc3, i_enc3)
        i_enc3 = self.I_LCA3(i_enc3, hv_enc3)
        
        # Level 4: Fourth downsampling (bottleneck)
        hv_enc4 = self.HVE_block4(hv_enc3)      # HV: (batch, 288, H/16, W/16)
        i_enc4 = self.IE_block4(i_enc3)         # I: (batch, 288, H/16, W/16)
        
        # Cross-attention at bottleneck
        hv_bottleneck = self.HV_LCA4(hv_enc4, i_enc4)
        i_bottleneck = self.I_LCA4(i_enc4, hv_enc4)
        
        
        # Bottleneck cross-attention
        hv_dec4 = self.HV_LCA5(hv_bottleneck, i_bottleneck)
        i_dec4 = self.I_LCA5(i_bottleneck, hv_bottleneck)
        
        # Level 4: First upsampling
        hv_dec3 = self.HVD_block4(hv_dec4, hv_enc3)    # With skip connection
        i_dec3 = self.ID_block4(i_dec4, i_enc3)        # With skip connection
        
        # Cross-attention at level 3
        hv_dec3 = self.HV_LCA6(hv_dec3, i_dec3)
        i_dec3 = self.I_LCA6(i_dec3, hv_dec3)
        
        # Level 3: Second upsampling
        hv_dec2 = self.HVD_block3(hv_dec3, hv_enc2)    # With skip connection
        i_dec2 = self.ID_block3(i_dec3, i_enc2)        # With skip connection
        
        # Cross-attention at level 2
        hv_dec2 = self.HV_LCA7(hv_dec2, i_dec2)
        i_dec2 = self.I_LCA7(i_dec2, hv_dec2)
        
        # Level 2: Third upsampling
        hv_dec1 = self.HVD_block2(hv_dec2, hv_enc1)    # With skip connection
        i_dec1 = self.ID_block2(i_dec2, i_enc1)        # With skip connection
        
        # Cross-attention at level 1
        hv_dec1 = self.HV_LCA8(hv_dec1, i_dec1)
        i_dec1 = self.I_LCA8(i_dec1, hv_dec1)
        
        # Level 1: Fourth upsampling
        hv_dec0 = self.HVD_block1(hv_dec1, hv_enc0)    # With skip connection
        i_dec0 = self.ID_block1(i_dec1, i_enc0)        # With skip connection
        
        # Level 0: Final processing
        hv_output = self.HVD_block0(hv_dec0)            # HV: (batch, 2, H, W)
        i_output = self.ID_block0(i_dec0)               # I: (batch, 1, H, W)
        
        
        # Combining both HV and I 
        output_hvi = torch.cat([hv_output, i_output], dim=1)  # (batch, 3, H, W)
        
        output_hvi = output_hvi + hvi        
        output_rgb = self.trans.PHVIT(output_hvi)
        
        return output_rgb


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

