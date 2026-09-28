import os
import time
import torch
import datetime
import argparse
import numpy as np
import hdf5storage
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import shutil
import random

# 导入自定义模块
from utils.getdataset import TrainDataset_V1, ValidDataset_V1
from utils.my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR, Loss_TV, Loss_MRAE, Loss_SAM
from utils.DataProcess import Data_Process
from architecture import model_generator

# 参数设置
parser = argparse.ArgumentParser(description="UM-SRNet Model Training")

parser.add_argument('--sh_name', type=str, default='train.sh', help='Shell script file name')
parser.add_argument('--py_name', type=str, default='train.py', help='Python script file name')

# GPU settings
parser.add_argument('--device', default='0', help='CUDA device')
parser.add_argument("--method", type=str, default='UM_SRNet', help='Model name')

# Directory settings
parser.add_argument('--mask_dir', type=str, default='./masks/train', help='Path to training mask directory')
parser.add_argument('--test_mask_dir', type=str, default=None, help='Path to test mask directory')
parser.add_argument('--train_data_path', type=str, default='./Data/train/', help='Path to training data')
parser.add_argument('--val_data_path', type=str, default='./Data/val/', help='Path to validation data')
parser.add_argument('--model_dir', default='model_UM_SRNet', help='Directory for saving models')

# Mask cropping settings
parser.add_argument('--start_dir', type=int, nargs=2, default=(0, 0), help='Starting coordinates for mask cropping')

# Training settings
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--end_epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--epoch_sam_num', type=int, default=2286, help='Number of samples per epoch')
parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")

# Model settings
parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
parser.add_argument('--out_channels', type=int, default=61, help='Number of output channels')
parser.add_argument('--dim', type=int, default=32, help='Feature dimension')
parser.add_argument('--deep_stage', type=int, default=3, help='Network depth')
parser.add_argument('--num_blocks', type=str, default='1,1,1,1', help='Number of blocks at each stage')
parser.add_argument('--num_heads', type=str, default='1,2,4,8', help='Number of attention heads at each stage')

# Image and patch settings
parser.add_argument('--image_size', type=int, nargs=2, default=(2048, 2048), help='Image size')
parser.add_argument('--train_patch_size', type=int, nargs=2, default=(512, 512), help='Training patch size')
parser.add_argument('--valid_patch_size', type=int, nargs=2, default=(512, 512), help='Validation patch size')
parser.add_argument('--sigma', type=float, default=(0, 1 / 255, 2 / 255, 3 / 255), help='Standard deviation of Gaussian noise')
args = parser.parse_args()

def load_masks(mask_dir):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.mat')]
    masks = []
    
    for mask_file in mask_files:
        mask_name = os.path.splitext(mask_file)[0]  
        mask_path = os.path.join(mask_dir, mask_file)
        mask_init = hdf5storage.loadmat(mask_path)['mask']
        mask = mask_init[:, args.start_dir[0]:args.start_dir[0]+args.image_size[0], 
                         args.start_dir[1]:args.start_dir[1]+args.image_size[1]]
        mask = np.clip(mask, 0, 1)
        mask = torch.from_numpy(mask).cuda()

        masks.append((mask_name, mask)) 
    
    return masks


os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_mrae = Loss_MRAE()
criterion_sam = Loss_SAM()
criterion_tv = Loss_TV(TVLoss_weight=float(0.5))

data_processing = Data_Process()

class LossIsNaN(Exception):
    pass

def save(psnr_mean, psnr_max, psnr_set, epoch, iteration, model_path, model, optimizer, scheduler):
    print('epoch %d, save checkpoint' % epoch)

    if psnr_mean >= psnr_max:
        psnr_max = psnr_mean
        if psnr_mean > psnr_set:
            save_checkpoint(model_path, epoch, iteration, model, optimizer, scheduler)

    return psnr_max

def train(model, train_set, iteration, total_iteration, per_epoch_iteration, optimizer, scheduler, patch_size, logger, masks):
    model.train()
    epoch_loss = 0.0
    global_item_num = 0

    epoch = iteration // per_epoch_iteration + 1
    
    begin = time.time()
    print(f'--epoch{epoch} start--')
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    
    train_bar = tqdm(train_loader, desc=f'Training Epoch [{epoch}]')

    for i, HSIs in enumerate(train_bar):
        
        HSIs = HSIs.cuda()
        
        current_batch_size = HSIs.size(0)
        
        mask_idx = random.randint(0, len(masks) - 1)
        mask_name, mask = masks[mask_idx]
        
        mask_patch = data_processing.get_random_mask_patches(
            mask=mask, 
            image_size=args.image_size, 
            patch_size=patch_size, 
            batch_size=current_batch_size
        )
        
        inputs, targets = data_processing.get_mos_hsi(
            hsi=HSIs, 
            mask=mask_patch, 
            sigma=args.sigma, 
            mos_size=patch_size[0], 
            hsi_input_size=patch_size[0], 
            hsi_target_size=patch_size[0]
        )

        lr = optimizer.param_groups[0]['lr']
        
        outputs = model(inputs, mask_patch)
        
        loss_rmse = criterion_rmse(outputs, targets)
        loss_tv = criterion_tv(outputs, targets) 
        loss_mrae = criterion_mrae(outputs, targets) * 0.2
        loss = loss_rmse + loss_tv + loss_mrae
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step() 
        
        epoch_loss += loss.item()
        global_item_num += 1
        iteration = iteration + 1
        
        train_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg Loss': f'{epoch_loss / global_item_num:.6f}'
        })
    
    end = time.time()
    
    epoch_loss = epoch_loss / global_item_num
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("{} - Iter[{:06d}/{:06d}], Epoch[{:06d}], Time[{:06d}], learning rate :{:.9f}, Train Loss: {:.9f}"
                .format(current_time, iteration, total_iteration, epoch, int((end - begin) / 60.), lr, epoch_loss))
    
    return epoch_loss, epoch, iteration


def validate_on_masks(val_loader, model, masks, logger, mask_type):

    mask_metrics = {
        'rmse': [],
        'psnr': [],
        'mrae': [],
        'sam': []
    }
    
    # for mask_idx, mask in enumerate(masks):
    for mask_idx, (mask_name, mask) in enumerate(masks):
        mask_rmse = AverageMeter()
        mask_psnr = AverageMeter()
        mask_mrae = AverageMeter()
        mask_sam = AverageMeter()
        
        val_bar = tqdm(val_loader, desc=f'Validating {mask_type} #{mask_name}')
        
        for i, HSIs in enumerate(val_bar):
            HSIs = HSIs.cuda()
            
            current_batch_size = HSIs.size(0)
            
            mask_patch = data_processing.get_random_mask_patches(
                mask=mask, 
                image_size=args.image_size, 
                patch_size=args.valid_patch_size, 
                batch_size=current_batch_size
            )
            
            inputs, targets = data_processing.get_mos_hsi(
                hsi=HSIs, 
                mask=mask_patch, 
                sigma=args.sigma, 
                mos_size=args.valid_patch_size[0], 
                hsi_input_size=args.valid_patch_size[0], 
                hsi_target_size=args.valid_patch_size[0]
            )
           
            with torch.no_grad():
                outputs = model(inputs, mask_patch)
                
                loss_rmse = criterion_rmse(outputs, targets)
                loss_psnr = criterion_psnr(outputs, targets)
                loss_mrae = criterion_mrae(outputs, targets)
                loss_sam = criterion_sam(outputs, targets)
                
                mask_rmse.update(loss_rmse.data)
                mask_psnr.update(loss_psnr.data)
                mask_mrae.update(loss_mrae.data)
                mask_sam.update(loss_sam.data)
            
            val_bar.set_postfix({
                'RMSE': f'{mask_rmse.avg:.6f}',
                'PSNR': f'{mask_psnr.avg:.6f}'
            })
        
        mask_metrics['rmse'].append(mask_rmse.avg)
        mask_metrics['psnr'].append(mask_psnr.avg)
        mask_metrics['mrae'].append(mask_mrae.avg)
        mask_metrics['sam'].append(mask_sam.avg)
    
    mask_metrics['avg_rmse'] = sum(mask_metrics['rmse']) / len(mask_metrics['rmse'])
    mask_metrics['avg_psnr'] = sum(mask_metrics['psnr']) / len(mask_metrics['psnr'])
    mask_metrics['avg_mrae'] = sum(mask_metrics['mrae']) / len(mask_metrics['mrae'])
    mask_metrics['avg_sam'] = sum(mask_metrics['sam']) / len(mask_metrics['sam'])
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(
        '{} - {} Average: RMSE = {:.9f}, PSNR = {:.9f}, MRAE = {:.9f}, SAM = {:.9f}'.format(
            current_time, mask_type, mask_metrics['avg_rmse'], mask_metrics['avg_psnr'], 
            mask_metrics['avg_mrae'], mask_metrics['avg_sam']))
    
    return mask_metrics

def validate(val_set, logger, model, train_masks, test_masks=None):

    model.eval()
    
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, drop_last=False)
    
    train_metrics = validate_on_masks(val_loader, model, train_masks, logger, "Train Mask")
    
    test_metrics = None
    if test_masks is not None:
        test_metrics = validate_on_masks(val_loader, model, test_masks, logger, "Test Mask")
    
    return train_metrics['avg_rmse'], train_metrics['avg_psnr'], train_metrics['avg_mrae'], train_metrics['avg_sam']


def main():
    global model, train_masks, test_masks
    
    model_path = args.model_dir
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_sh_path = os.path.join(script_dir, args.sh_name)
    if os.path.exists(train_sh_path):
        shutil.copy(train_sh_path, os.path.join(model_path, args.sh_name))
        print(f"train.sh copied into {model_path}")

    train_py_path = os.path.join(script_dir, args.py_name)
    if os.path.exists(train_py_path):
        shutil.copy(train_py_path, os.path.join(model_path, args.py_name))
        print(f"train.py have copied into {model_path}")

    logger = initialize_logger(os.path.join(model_path, 'train.log'))
    
    print("loading mask...")
    train_masks = load_masks(args.mask_dir)
    
    test_masks = None
    if args.test_mask_dir and os.path.exists(args.test_mask_dir):
        test_masks = load_masks(args.test_mask_dir)
    
    print("\nloading datasets...")
    train_set = TrainDataset_V1(data_path=args.train_data_path, patch_size=args.train_patch_size, arg=True)
    val_set = ValidDataset_V1(data_path=args.val_data_path, patch_size=args.valid_patch_size, arg=True)
    print('TrainDataset:', len(train_set))
    print('ValidDataset:', len(val_set))
    
    num_blocks = [int(x) for x in args.num_blocks.split(',')]
    num_heads = [int(x) for x in args.num_heads.split(',')]
    
    model = model_generator(args.method, args.pretrained_model_path)

    if torch.cuda.is_available():
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_tv.cuda()
        criterion_mrae.cuda()
    
    iteration = 0
    psnr_max = 0

    per_epoch_iteration = args.epoch_sam_num // args.batch_size
    total_iteration = per_epoch_iteration*args.end_epoch

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    # -------------------------
    # Resume training if needed
    # -------------------------
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        iteration = checkpoint['iter']
        print(f"=> Resumed from iteration {iteration}")

        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Scheduler state aligned.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    while iteration < total_iteration:

        loss, epoch, iteration = train(
            model,
            train_set,
            iteration,
            total_iteration,
            per_epoch_iteration,
            optimizer,
            scheduler,
            args.train_patch_size,
            logger=logger,
            masks=train_masks
        )

        rmse_loss, psnr_loss, mrae_loss, sam_loss = validate(
            val_set,
            logger,
            model,
            train_masks,
            test_masks
        )

        psnr_max = save(psnr_loss, psnr_max, 10, epoch, iteration, model_path, model, optimizer, scheduler)

if __name__ == '__main__':
    main()
