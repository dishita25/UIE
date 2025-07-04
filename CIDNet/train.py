import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    loss_rgb_print = 0  # For rgb loss tracking
    loss_hvi_print = 0  # For hvi loss tracking
    pic_cnt = 0
    loss_last_10 = 0
    loss_rgb_last_10 = 0  # This as well
    loss_hvi_last_10 = 0  # This as well
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)  
            
        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_rgb_print = loss_rgb_print + loss_rgb.item()  # This
        loss_hvi_print = loss_hvi_print + loss_hvi.item()  # This
        loss_last_10 = loss_last_10 + loss.item()
        loss_rgb_last_10 = loss_rgb_last_10 + loss_rgb.item()  # This
        loss_hvi_last_10 = loss_hvi_last_10 + loss_hvi.item()  # This
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Total Loss: {:.4f} | RGB Loss: {:.4f} | HVI Loss: {:.4f} || Learning rate: lr={}.".format(
            epoch, loss_last_10/pic_last_10, loss_rgb_last_10/pic_last_10, 
            loss_hvi_last_10/pic_last_10, optimizer.param_groups[0]['lr']))  # Changed this print
            
            loss_last_10 = 0
            loss_rgb_last_10 = 0  # This
            loss_hvi_last_10 = 0  # This
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, loss_rgb_print, loss_hvi_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_fivek_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    
    elif opt.EUVP:
        train_set = get_EUVP_training_set(opt.data_train_EUVP, size = opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_EUVP)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, epoch_rgb_loss, epoch_hvi_loss, pic_num = train(epoch)  # Updated
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.fivek:
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            if opt.EUVP:
                output_folder = 'EUVP/'
                label_dir = opt.data_valgt_EUVP
                norm_size = True
            
            im_dir = opt.val_folder + output_folder + '*.jpg'
            
            # Validation set evaluation
            print("\n===> Evaluating on Validation Set:")
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            # # To check what file format the evaluation is actually saving
            # import os
            # print(f"Checking output directory: {opt.val_folder+output_folder}")
            # if os.path.exists(opt.val_folder+output_folder):
            #     files = os.listdir(opt.val_folder+output_folder)
            #     print(f"Files found: {files}")
            #     print(f"File extensions: {[f.split('.')[-1] for f in files if '.' in f]}")
            # else:
            #     print("Output directory doesn't exist!")

            
            avg_psnr_val, avg_ssim_val, avg_lpips_val = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Validation - Avg.PSNR: {:.4f} dB ".format(avg_psnr_val))
            print("===> Validation - Avg.SSIM: {:.4f} ".format(avg_ssim_val))
            print("===> Validation - Avg.LPIPS: {:.4f} ".format(avg_lpips_val))
            
            # Training set evaluation - ADD THIS SECTION
            print("\n===> Evaluating on Training Set:")
            train_output_folder = opt.val_folder + output_folder + 'train/'
            if not os.path.exists(train_output_folder):
                os.makedirs(train_output_folder)
                
            # Create a separate data loader for training set evaluation with batch_size=1
            if opt.lol_v1:
                train_eval_set = get_eval_set(opt.data_train_lol_v1 + '/low')  # Use low folder
            elif opt.lolv2_real:
                train_eval_set = get_eval_set(opt.data_train_lolv2_real + '/Low')  # Use Low folder
            elif opt.lolv2_syn:
                train_eval_set = get_eval_set(opt.data_train_lolv2_syn + '/Low')  # Use Low folder
            elif opt.EUVP:
                train_eval_set = get_eval_set(opt.data_train_EUVP + '/trainA')  # Use trainA folder
            else:
                print("Error: Unsupported dataset for training evaluation")
                continue
            
            # Create evaluation data loader with batch_size=1
            training_eval_loader = DataLoader(dataset=train_eval_set, num_workers=1, batch_size=1, shuffle=False)

            eval(model, training_eval_loader, model_out_path, train_output_folder,
                norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)

           
            train_im_dir = train_output_folder + '*.jpg'
            
            # Determine training ground truth directory based on dataset
            if opt.lol_v1:
                train_label_dir = opt.data_train_lol_v1 + '/high/'
            elif opt.lolv2_real:
                train_label_dir = opt.data_train_lolv2_real + '/Normal/'
            elif opt.lolv2_syn:
                train_label_dir = opt.data_train_lolv2_syn + '/Normal/'
            elif opt.EUVP:
                train_label_dir = opt.data_train_EUVP + '/trainB/'
            else:
                train_label_dir = label_dir  # Fallback
                
            avg_psnr_train, avg_ssim_train, avg_lpips_train = metrics(train_im_dir, train_label_dir, use_GT_mean=False)
            
            print("===> Training - Avg.PSNR: {:.4f} dB ".format(avg_psnr_train))
            print("===> Training - Avg.SSIM: {:.4f} ".format(avg_ssim_train))
            print("===> Training - Avg.LPIPS: {:.4f} ".format(avg_lpips_train))
            
            # Store both validation and training metrics
            psnr.append({'val': avg_psnr_val, 'train': avg_psnr_train})
            ssim.append({'val': avg_ssim_val, 'train': avg_ssim_train})
            lpips.append({'val': avg_lpips_val, 'train': avg_lpips_train})
            
            print("Validation metrics:", [p['val'] for p in psnr])
            print("Training metrics:", [p['train'] for p in psnr])
           
        torch.cuda.empty_cache()
    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("| Epochs | Val_PSNR | Val_SSIM | Val_LPIPS | Train_PSNR | Train_SSIM | Train_LPIPS |\n")
        f.write("|--------|----------|----------|-----------|-------------|-------------|-------------|\n")
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | {psnr[i]['val']:.4f} | {ssim[i]['val']:.4f} | {lpips[i]['val']:.4f} | {psnr[i]['train']:.4f} | {ssim[i]['train']:.4f} | {lpips[i]['train']:.4f} |\n")  
            