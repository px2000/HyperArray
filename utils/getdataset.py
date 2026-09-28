from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import random
import os
import hdf5storage
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
import torch.nn.functional as F
import h5py

class MaskDataset(Dataset):
    def __init__(self, mask_dir):
        self.mask_dir = mask_dir
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.mat')]
        self.mask_files.sort()

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_file = self.mask_files[idx]
        mask_name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(self.mask_dir, mask_file)

        mask_init = hdf5storage.loadmat(mask_path)['mask']  # numpy array [C,H,W]
        mask = mask_init[:, args.start_dir[0]:args.start_dir[0]+args.image_size[0], 
                         args.start_dir[1]:args.start_dir[1]+args.image_size[1]]
        mask = np.clip(mask, 0, 1)
        mask = torch.from_numpy(mask)

        return mask_name, mask

class TrainDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()
        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0,hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0,hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()
        
        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_paths[idx], 'r')
        hsi = f['hsi'][:]
        f.close()
    
        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class TestDataset_V1(Dataset):
    def __init__(self, data_path, patch_size):
        self.data_paths = []
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        for file_name in data_list:
            self.data_paths.append(os.path.join(data_path, file_name))

        self.img_num = len(self.data_paths)

    def __getitem__(self, idx):
        with h5py.File(self.data_paths[idx], 'r') as f:
            hsi = f['hsi'][:]

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        _, h, w = hsi.shape

        if patch_size_h > h or patch_size_w > w:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than "
                f"HSI spatial size {(h, w)}: {self.data_paths[idx]}"
            )

        start_h = (h - patch_size_h) // 2
        start_w = (w - patch_size_w) // 2

        output_hsi = hsi[
            :,
            start_h:start_h + patch_size_h,
            start_w:start_w + patch_size_w
        ]

        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class TestDataset_MOS(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            print('data_list[i]', self.data_list[i])
            print('data_path', self.data_path)
            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]
            bmp = bmp[self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            bmp = bmp / bmp.max()
            bmp = bmp.astype(np.float32)
            mos = np.expand_dims(bmp, axis=0)
            self.MOS_list.append(mos)
            
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num

