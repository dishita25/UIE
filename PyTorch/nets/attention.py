
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioChannelAttention(nn.Module):
   
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatioChannelAttention, self).__init__()
        
        self.conv_before_attention = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        
        # Channel Attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial Attention branch
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True) 
        self.spatial_sigmoid = nn.Sigmoid()

        # Convolution after concatenation of channel and spatial outputs
        self.combine_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()

        # Apply convolution before attention
        x_conv = self.conv_before_attention(x)
        
        # Channel Attention branch
        avg_pool = self.avg_pool(x_conv)                            # b x c x 1 x 1
        channel_att = self.channel_excitation(avg_pool)             # b x c x 1 x 1
        channel_feature = x_conv * channel_att                           # Channel scaled feature map

        # Spatial Attention branch
        avg_out = torch.mean(x_conv, dim=1, keepdim=True)                                # b x 1 x H x W
        max_out, _ = torch.max(x_conv, dim=1, keepdim=True)                              # b x 1 x H x W
        spatial_descriptor = torch.cat([avg_out, max_out], dim=1)                   # b x 2 x H x W
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_descriptor))   # b x 1 x H x W
        spatial_feature = x_conv * spatial_att                                           # Spatial scaled feature map

        # Concatenate spatial and channel attention refined features
        combined = torch.cat([spatial_feature, channel_feature], dim=1)             # b x 2c x H x W

        # Apply convolution on concatenated features
        combined_conv = self.combine_conv(combined)         # b x c x H x W

        # Add original input element-wise (residual connection)
        out = combined_conv + x
        return out
