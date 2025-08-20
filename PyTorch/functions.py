# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import torch.nn as nn
# import scipy.io as sio
# import math
# from skimage import io as img
# from skimage import color, morphology, filters
# from imresize import imresize
# import os
# import random
# from sklearn.cluster import KMeans
# from imresize import imresize
# import torch.nn.functional as F

# # custom weights initialization called on netG and netD

# def read_image(opt):
#     x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
#     x = np2torch(x,opt)
#     x = x[:,0:3,:,:]
#     return x

# def denorm(x):
#     # out = (x + 1) / 2
#     return x.clamp(0, 1)

# def norm(x):
#     # out = (x -0.5) *2
#     return x.clamp(0, 1)

# def convert_image_np(inp):
#     if inp.shape[1]==3:
#         inp = denorm(inp)
#         inp = move_to_cpu(inp[-1,:,:,:])
#         inp = inp.numpy().transpose((1,2,0))
#     else:
#         inp = denorm(inp)
#         inp = move_to_cpu(inp[-1,-1,:,:])
#         inp = inp.numpy().transpose((0,1))
#     inp = np.clip(inp,0,1)
#     return inp

# def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
#     fig,ax = plt.subplots(1)
#     if ncs==1:
#         ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
#     else:
#         ax.imshow(convert_image_np(real_cpu.cpu()))
#     rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     ax.axis('off')
#     plt.savefig(file_name)
#     plt.close(fig)

# def convert_image_np_2d(inp):
#     inp = denorm(inp)
#     inp = inp.numpy()
#     return inp 

# def generate_blur_input(size, num_samp=1, device='cuda', blur_image_path=None, scale=1):
#     """
#     Generate blurry image input instead of noise
#     """
#     if blur_image_path is None or not os.path.exists(blur_image_path):
#         print(f"Warning: Blur image path '{blur_image_path}' is invalid or not provided. Falling back to noise generation.")
#         return generate_noise(size, num_samp, device, 'gaussian', scale)
    
#     try:
#         blur_img = img.imread(blur_image_path)
        
#         if len(blur_img.shape) == 3 and blur_img.shape[2] >= 3:
#             blur_img = blur_img[:, :, :3]
#         elif len(blur_img.shape) == 2:
#             blur_img = np.stack([blur_img] * 3, axis=2)
#         else:
#             print(f"Warning: Unsupported image format for {blur_image_path}. Falling back to noise.")
#             return generate_noise(size, num_samp, device, 'gaussian', scale)
            
#         blur_img = blur_img.astype(np.float32) / 255.0
        
#         blur_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1)).unsqueeze(0)
#         blur_tensor = blur_tensor.to(device)
        
#         target_c, target_h, target_w = size
        
#         blur_tensor = F.interpolate(blur_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

#         if num_samp > 1:
#             blur_tensor = blur_tensor.expand(num_samp, -1, -1, -1)
        
#         if blur_tensor.shape[1] != target_c:
#             if target_c == 1 and blur_tensor.shape[1] == 3:
#                 blur_tensor = torch.mean(blur_tensor, dim=1, keepdim=True)
#             elif target_c == 3 and blur_tensor.shape[1] == 1:
#                 blur_tensor = blur_tensor.expand(-1, 3, -1, -1)
#             else:
#                 print(f"Warning: Channel mismatch (image has {blur_tensor.shape[1]}, target {target_c}). Falling back to noise.")
#                 return generate_noise(size, num_samp, device, 'gaussian', scale)
        
#         blur_tensor = norm(blur_tensor)
#         return blur_tensor
        
#     except Exception as e:
#         print(f"Error loading or processing blur image from {blur_image_path}: {e}")
#         print("Falling back to noise generation")
#         return generate_noise(size, num_samp, device, 'gaussian', scale)

# def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
#     if type == 'gaussian':
#         noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
#         noise = upsampling(noise,size[1], size[2])
#         print(f"Shape: {noise.shape}")
#     if type =='gaussian_mixture':
#         noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
#         noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
#         noise = noise1+noise2
#     if type == 'uniform':
#         noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
#     return noise

# def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
#     fig,ax = plt.subplots(1)
#     n = np.arange(0,epochs)
#     plt.plot(n,G_loss,n,D_loss)
#     plt.xlabel('epochs')
#     plt.legend([label1,label2],loc='upper right')
#     plt.savefig('%s.png' % name)
#     plt.close(fig)

# def plot_learning_curve(loss,epochs,name):
#     fig,ax = plt.subplots(1)
#     n = np.arange(0,epochs)
#     plt.plot(n,loss)
#     plt.ylabel('loss')
#     plt.xlabel('epochs')
#     plt.savefig('%s.png' % name)
#     plt.close(fig)

# def upsampling(im,sx,sy):
#     m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
#     return m(im)

# def reset_grads(model,require_grad):
#     for p in model.parameters():
#         p.requires_grad_(require_grad)
#     return model

# def move_to_gpu(t):
#     if (torch.cuda.is_available()):
#         t = t.to(torch.device('cuda'))
#     return t

# def move_to_cpu(t):
#     t = t.to(torch.device('cpu'))
#     return t

# def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
#     if real_data.shape[2:] != fake_data.shape[2:]:
#         h = min(real_data.shape[2], fake_data.shape[2])
#         w = min(real_data.shape[3], fake_data.shape[3])
#         real_data = real_data[:, :, :h, :w]
#         fake_data = fake_data[:, :, :h, :w]

#     alpha = torch.rand(1, 1, 1, 1, device=device)
#     alpha = alpha.expand_as(real_data)

#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#     interpolates.requires_grad_(True)

#     disc_interpolates = netD(interpolates, real_data)
    
#     gradients = torch.autograd.grad(
#         outputs=disc_interpolates, inputs=interpolates,
#         grad_outputs=torch.ones_like(disc_interpolates),
#         create_graph=True, retain_graph=True, only_inputs=True
#     )[0]

#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#     return gradient_penalty
    
# def read_image_dir(dir,opt):
#     x = img.imread('%s' % (dir))
#     x = np2torch(x,opt)
#     x = x[:,0:3,:,:]
#     return x

# def read_blur_image(opt):
#     x  = img.imread('%s' % (opt.blur_image_path))
#     x = np2torch(x,opt)
#     x = x[0:,0:3,:,:]
#     return x

# def np2torch(x,opt):
#     if opt.nc_im == 3:
#         x = x[:,:,:,None]
#         x = x.transpose((3, 2, 0, 1))/255
#     else:
#         x = color.rgb2gray(x)
#         x = x[:,:,None,None]
#         x = x.transpose(3, 2, 0, 1)
#     x = torch.from_numpy(x)
#     if not(opt.not_cuda):
#         x = move_to_gpu(x)
#     x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
#     x = norm(x)
#     return x

# def torch2uint8(x):
#     x = x[0,:,:,:]
#     x = x.permute((1,2,0))
#     x = 255*denorm(x)
#     x = x.cpu().numpy()
#     x = x.astype(np.uint8)
#     return x

# def read_image2np(opt):
#     x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
#     x = x[:, :, 0:3]
#     return x

# def save_networks(netG,netD,z,opt):
#     torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
#     torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
#     torch.save(z, '%s/z_opt.pth' % (opt.outf))

# def generate_dir2save(opt):
#     dir2save = None
#     if (opt.mode == 'train') | (opt.mode == 'SR_train'):
#         dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
#     elif (opt.mode == 'animation_train') :
#         dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
#     elif (opt.mode == 'paint_train') :
#         dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
#     elif opt.mode == 'random_samples':
#         dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
#     elif opt.mode == 'random_samples_arbitrary_sizes':
#         dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
#     elif opt.mode == 'animation':
#         dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
#     elif opt.mode == 'SR':
#         dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
#     elif opt.mode == 'harmonization':
#         dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
#     elif opt.mode == 'editing':
#         dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
#     elif opt.mode == 'paint2image':
#         dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
#         if opt.quantization_flag:
#             dir2save = '%s_quantized' % dir2save
#     return dir2save


# def post_config(opt):
#     opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
#     opt.niter_init = opt.niter
#     opt.noise_amp_init = opt.noise_amp
#     opt.nfc_init = opt.nfc
#     opt.min_nfc_init = opt.min_nfc
#     opt.scale_factor_init = opt.scale_factor_init
#     opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor_init)
    
#     if not hasattr(opt, 'blur_image_path') or opt.blur_image_path is None:
#         opt.blur_image_path = "/path/to/your/blurry_image.jpg" 
#         print(f"Warning: 'blur_image_path' not set. Using default: {opt.blur_image_path}")
    
#     if not os.path.exists(opt.blur_image_path):
#         print(f"Error: Blurry image not found at '{opt.blur_image_path}'. Please provide a valid path.")
    
#     if opt.mode == 'SR':
#         opt.alpha = 100

#     if opt.manualSeed is None:
#         opt.manualSeed = random.randint(1, 10000)
#     print("Random Seed: ", opt.manualSeed)
#     random.seed(opt.manualSeed)
#     torch.manual_seed(opt.manualSeed)
#     if torch.cuda.is_available() and opt.not_cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     return opt

# # NEW function to create the pyramid with hard-coded scales
# def creat_pyramid_from_hardcoded_scales(real, scales):
#     reals = []
#     real = real[:, 0:3, :, :]
#     for h, w in scales:
#         curr_real = torch.nn.functional.interpolate(real, size=(h, w), mode='bilinear', align_corners=False)
#         reals.append(curr_real)
#     return reals

# # MODIFIED draw_concat to use hard-coded scales
# def draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, hardcoded_scales):
#     G_z = in_s
#     if len(Gs) > 0:
#         for idx, (G, Z_opt, blur_curr, noise_amp) in enumerate(zip(Gs, Zs, blurs, NoiseAmp)):
#             print(f"Z_opt shape: {Z_opt.shape}, G_z shape before upscaling: {G_z.shape}, blur_curr shape: {blur_curr.shape}")
            
#             # Resize G_z to match the current blur_curr shape
#             G_z = torch.nn.functional.interpolate(G_z, size=(blur_curr.shape[2], blur_curr.shape[3]), mode='bilinear', align_corners=False)
#             print(f"G_z shape after resizing to match the current scale: {G_z.shape}")
            
#             G_z = m_image(G_z)
#             print(f"G_z shape after padding: {G_z.shape}")

#             z = torch.zeros_like(Z_opt, device=opt.device)
#             z, G_z = align_tensors(z, G_z)
#             z_in = G_z + noise_amp * z
#             print(f"z_in shape: {z_in.shape}")     

#             G_z = G(z_in.detach())
#             print(f"G_z shape after detach function: {G_z.shape}")

#             # Upscale for next scale using the hard-coded sizes
#             if idx < len(hardcoded_scales) - 1:
#                 target_h, target_w = hardcoded_scales[idx + 1]
#                 G_z = torch.nn.functional.interpolate(G_z, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
#             print(f"G_z shape after imresize: {G_z.shape}")

#     return G_z


# def align_tensors(a, b):
#     h = min(a.shape[2], b.shape[2])
#     w = min(a.shape[3], b.shape[3])
#     return a[:, :, :h, :w], b[:, :, :h, :w]


#NEW CODE FOR FEW SHOT LEARNING
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from imresize import imresize
import os
import random
from sklearn.cluster import KMeans
from imresize import imresize
import torch.nn.functional as F

# custom weights initialization called on netG and netD

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def denorm(x):
    # out = (x + 1) / 2
    return x.clamp(0, 1)

def norm(x):
    # out = (x -0.5) *2
    return x.clamp(0, 1)

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    return inp 

def generate_noise_batch(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    if real_data.shape[2:] != fake_data.shape[2:]:
        h = min(real_data.shape[2], fake_data.shape[2])
        w = min(real_data.shape[3], fake_data.shape[3])
        real_data = real_data[:, :, :h, :w]
        fake_data = fake_data[:, :, :h, :w]

    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, real_data)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
    
def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image_dir(dir, opt):
    x = img.imread('%s' % (dir))
    # `np2torch` now returns a 3D tensor (C, H, W)
    x = np2torch(x, opt)
    # This slicing is no longer necessary and should be removed.
    # The image is already in the correct format.
    # The line "x = x[:,0:3,:,:]" should be removed or commented out.
    return x

def np2torch(x, opt):
    if opt.nc_im == 3:
        if x.ndim == 2:
            # Convert grayscale to color if needed
            x = np.stack([x] * 3, axis=-1)
        # Transpose from (H, W, C) to (C, H, W)
        x = x.transpose((2, 0, 1)) / 255.0
    else:
        # For grayscale images
        x = color.rgb2gray(x)
        # Add a channel dimension for grayscale
        x = x[None, :, :]
    
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def generate_dir2save(opt):
    dir2save = None
    dir2save = 'TrainedModels/%s/few_shot_training' % (opt.real_dir.split('/')[-2])
    return dir2save

def post_config(opt):
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor_init
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor_init)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def creat_pyramid_from_hardcoded_scales_batch(real_batch, scales):
    reals = []
    print(f"Input batch shape: {real_batch.shape}")
    real_batch = real_batch[:, 0:3, :, :]
    print(f"Before interpolation: {real_batch.shape}") # keeps all batch elements
    for h, w in scales:
        # Interpolate the entire batch
        curr_real = torch.nn.functional.interpolate(
            real_batch, size=(h, w), mode='bilinear', align_corners=False
        )
        print(f"Interpolated shape for scale ({h}, {w}): {curr_real.shape}")
        reals.append(curr_real)
    return reals

def draw_concat_hardcoded_batch(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, hardcoded_scales, batch_size, last_scale_idx=None):
    G_z = in_s
    if last_scale_idx is None:
        last_scale_idx = len(Gs) - 1

    if len(Gs) > 0:
        for idx, (G, Z_opt, blur_curr, noise_amp) in enumerate(
            zip(Gs[:last_scale_idx + 1], Zs[:last_scale_idx + 1], blurs[:last_scale_idx + 1], NoiseAmp[:last_scale_idx + 1])
        ):
            # Resize to current blur scale (keep batch dim)
            G_z = torch.nn.functional.interpolate(
                G_z, size=(blur_curr.shape[2], blur_curr.shape[3]),
                mode='bilinear', align_corners=False
            )

            # Apply mask/padding to each image in batch
            G_z = m_image(G_z)

            # Prepare latent noise for batch
            z = torch.zeros_like(Z_opt, device=opt.device)
            
            # Align shapes across batch
            z, G_z = align_tensors(z, G_z)

            # Inject noise
            z_in = G_z + noise_amp * z

            # Generate output for batch
            G_z = G(z_in.detach())

            # Upscale to next hardcoded scale if not at last
            if idx < len(hardcoded_scales) - 1:
                target_h, target_w = hardcoded_scales[idx + 1]
                G_z = torch.nn.functional.interpolate(
                    G_z, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )

    return G_z


# def creat_pyramid_from_hardcoded_scales(real, scales):
#     reals = []
#     real = real[:, 0:3, :, :]
#     for h, w in scales:
#         curr_real = torch.nn.functional.interpolate(real, size=(h, w), mode='bilinear', align_corners=False)
#         reals.append(curr_real)
#     return reals

# # MODIFIED draw_concat to use hard-coded scales
# def draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, hardcoded_scales):
#     G_z = in_s
#     if len(Gs) > 0:
#         for idx, (G, Z_opt, blur_curr, noise_amp) in enumerate(zip(Gs, Zs, blurs, NoiseAmp)):
#             print(f"Z_opt shape: {Z_opt.shape}, G_z shape before upscaling: {G_z.shape}, blur_curr shape: {blur_curr.shape}")
            
#             # Resize G_z to match the current blur_curr shape
#             G_z = torch.nn.functional.interpolate(G_z, size=(blur_curr.shape[2], blur_curr.shape[3]), mode='bilinear', align_corners=False)
#             print(f"G_z shape after resizing to match the current scale: {G_z.shape}")
            
#             G_z = m_image(G_z)
#             print(f"G_z shape after padding: {G_z.shape}")

#             z = torch.zeros_like(Z_opt, device=opt.device)
#             z, G_z = align_tensors(z, G_z)
#             z_in = G_z + noise_amp * z
#             print(f"z_in shape: {z_in.shape}")     

#             G_z = G(z_in.detach())
#             print(f"G_z shape after detach function: {G_z.shape}")

#             # Upscale for next scale using the hard-coded sizes
#             if idx < len(hardcoded_scales) - 1:
#                 target_h, target_w = hardcoded_scales[idx + 1]
#                 G_z = torch.nn.functional.interpolate(G_z, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
#             print(f"G_z shape after imresize: {G_z.shape}")

#     return G_z


def align_tensors(a, b):
    h = min(a.shape[2], b.shape[2])
    w = min(a.shape[3], b.shape[3])
    return a[:, :, :h, :w], b[:, :, :h, :w]
