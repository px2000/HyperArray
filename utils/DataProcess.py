import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

class Data_Process(object):
    def __init__(self):
        self.noise_sigma = 0
        self.hsi_max = []

    def add_noise(self, inputs, sigma):
        noise = torch.zeros_like(inputs)
        noise.normal_(0, sigma)
        noisy = inputs + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy
    
    
    def get_random_mask_patches(self, mask, image_size, patch_size, batch_size):
        if mask.dim() == 3:
            masks = []
            for i in range(batch_size):
                random_h = random.randint(0, image_size[0] - patch_size[0])
                random_w = random.randint(0, image_size[1] - patch_size[1])
                mask_patch = mask[:, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]
                masks.append(mask_patch)
        elif mask.dim() == 4:
            masks = []
            for i in range(batch_size):
                random_h = random.randint(0, image_size[0] - patch_size[0])
                random_w = random.randint(0, image_size[1] - patch_size[1])
                mask_patch = mask[i, :, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]] 
                masks.append(mask_patch)
            
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches


    def get_fix_mask_patches(self, mask, image_size, patch_size, batch_size):
        if mask.dim() == 3:
            masks = []

            start_h = (image_size[0] - patch_size[0]) // 2
            start_w = (image_size[1] - patch_size[1]) // 2

            for i in range(batch_size):
                mask_patch = mask[
                    :,
                    start_h:start_h + patch_size[0],
                    start_w:start_w + patch_size[1]
                ]
                masks.append(mask_patch)

        elif mask.dim() == 4:
            masks = []

            start_h = (image_size[0] - patch_size[0]) // 2
            start_w = (image_size[1] - patch_size[1]) // 2

            for i in range(batch_size):
                mask_patch = mask[
                    i,
                    :,
                    start_h:start_h + patch_size[0],
                    start_w:start_w + patch_size[1]
                ]
                masks.append(mask_patch)

        mask_patches = torch.stack(masks, dim=0)

        return mask_patches
            
        
    def get_mos_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=650, hsi_target_size=650, init_div_rat=8):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi

        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max

        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)

        return input_mos, output_hsi


    def extend_spatial_resolution(self, hsi, extend_rate):
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        return hsi_extend

