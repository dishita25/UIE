
import torch
import torch.nn as nn
import math

pi = math.pi

class RGB_HVI(nn.Module):
    def __init__(self, beta=0.3, gamma=0.05, alpha_s=1.3):
        super(RGB_HVI, self).__init__()
        self.density_k = nn.Parameter(torch.tensor([0.2]))  # learnable nonlinearity
        self.beta = beta
        self.gamma = gamma
        self.alpha_s = alpha_s
        self.gated = True
        self.this_k = 0

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype

        value = img.max(1)[0]
        img_min = img.min(1)[0]

        hue = torch.zeros_like(value)
        hue[img[:,2]==value] = 4.0 + ((img[:,0]-img[:,1])/(value-img_min+eps))[img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ((img[:,2]-img[:,0])/(value-img_min+eps))[img[:,1]==value]
        hue[img[:,0]==value] = ((img[:,1]-img[:,2])/(value-img_min+eps))[img[:,0]==value] % 6
        hue[img_min==value] = 0.0
        hue = hue / 6.0  # normalized [0,1]

        blue_mask = ((hue >= 0.55) & (hue <= 0.66)).float()
        hue = (hue + self.gamma * blue_mask) % 1.0

        saturation = (value - img_min) / (value + eps)
        saturation[value==0] = 0.0

        k = self.density_k
        self.this_k = k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        color_sensitive = color_sensitive * (1.0 + self.beta * blue_mask)

        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        xyz = torch.stack([H,V,I], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:,0,:,:], img[:,1,:,:], img[:,2,:,:]
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)

        k = self.this_k
        color_sensitive = ((I * 0.5 * pi).sin() + eps).pow(k)
        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)

        h = torch.atan2(V+eps, H+eps) / (2*pi) % 1.0
        s = torch.sqrt(H**2 + V**2 + eps)
        v = I

        if self.gated:
            s = s * self.alpha_s
        s = torch.clamp(s,0,1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h*6.0)
        f = h*6.0 - hi
        p = v * (1 - s)
        q = v * (1 - f*s)
        t = v * (1 - (1-f)*s)

        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5

        r[hi0]=v[hi0]; g[hi0]=t[hi0]; b[hi0]=p[hi0]
        r[hi1]=q[hi1]; g[hi1]=v[hi1]; b[hi1]=p[hi1]
        r[hi2]=p[hi2]; g[hi2]=v[hi2]; b[hi2]=t[hi2]
        r[hi3]=p[hi3]; g[hi3]=q[hi3]; b[hi3]=v[hi3]
        r[hi4]=t[hi4]; g[hi4]=p[hi4]; b[hi4]=v[hi4]
        r[hi5]=v[hi5]; g[hi5]=p[hi5]; b[hi5]=q[hi5]

        rgb = torch.stack([r,g,b], dim=1)
        return rgb
