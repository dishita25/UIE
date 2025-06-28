# train_funiegan_singan.py

import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from imresize import imresize
import functions  
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def train_single_image_with_funiegan(opt):
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = []
    reals = functions.creat_reals_pyramid(real, reals, opt)

    Gs, Zs, NoiseAmp = [], [], []
    in_s = torch.full_like(reals[0], 0, device=opt.device)

    for scale_num in range(opt.stop_scale + 1):
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
        os.makedirs(opt.outf, exist_ok=True)

        real = reals[scale_num]
        opt.nzx, opt.nzy = real.shape[2], real.shape[3]

        generator = GeneratorFunieGAN().to(opt.device)
        discriminator = DiscriminatorFunieGAN().to(opt.device)
        generator.apply(functions.weights_init)
        discriminator.apply(functions.weights_init)

        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        mse = nn.MSELoss().to(opt.device)
        l1 = nn.L1Loss().to(opt.device)
        adv_criterion = nn.MSELoss().to(opt.device)
        perceptual = VGG19_PercepLoss().to(opt.device)

        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_noise = nn.ZeroPad2d(pad_noise)
        m_image = nn.ZeroPad2d(pad_noise)

        fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
        z_opt = torch.full_like(fixed_noise, 0)
        z_opt = m_noise(z_opt)

        for epoch in range(opt.niter):
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

            if scale_num == 0:
                z_prev = torch.full_like(noise_, 0)
                prev = m_image(z_prev)
                opt.noise_amp = 1
            else:
                prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)
                z_prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                rmse = torch.sqrt(mse(real, z_prev))
                opt.noise_amp = opt.noise_amp_init * rmse
                z_prev = m_image(z_prev)

            noise = opt.noise_amp * noise_ + prev
            fake = generator(noise.detach())

            discriminator.zero_grad()
            real_pred = discriminator(real, real)
            fake_pred = discriminator(fake.detach(), real)
            loss_D = 0.5 * (adv_criterion(real_pred, torch.ones_like(real_pred)) +
                            adv_criterion(fake_pred, torch.zeros_like(fake_pred)))
            loss_D.backward()
            optimizer_D.step()

            generator.zero_grad()
            fake = generator(noise.detach())
            fake_pred = discriminator(fake, real)

            loss_adv = adv_criterion(fake_pred, torch.ones_like(fake_pred))
            loss_l1 = l1(fake, real)
            loss_vgg = perceptual(fake, real)
            loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
            loss_G.backward()
            optimizer_G.step()

            if epoch % 500 == 0 or epoch == opt.niter - 1:
                plt.imsave(f"{opt.outf}/fake_epoch_{epoch}.png", functions.convert_image_np(fake.detach()))

        Gs.append(generator.eval())
        Zs.append(z_opt)
        NoiseAmp.append(opt.noise_amp)

        torch.save(generator.state_dict(), f"{opt.outf}/netG.pth")
        torch.save(discriminator.state_dict(), f"{opt.outf}/netD.pth")
        torch.save(z_opt, f"{opt.outf}/z_opt.pth")

    return Gs, Zs, reals, NoiseAmp


if __name__ == '__main__':
    opt = get_config()
    train_single_image_with_funiegan(opt)
