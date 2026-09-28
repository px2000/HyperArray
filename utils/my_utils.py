from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer, scheduler):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label  + 1e-4) / (label + 1e-4)

        mrae = torch.mean(error)
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        num = torch.sum(outputs * labels, 1)
        den = torch.sqrt(torch.sum(outputs * outputs, 1)) * torch.sqrt(torch.sum(labels * labels, 1)) 
        sam = torch.arccos((num) / (den)).mean()
        return sam

class Loss_TV(nn.Module):
    def __init__(self, TVLoss_weight: float=1):
        super(Loss_TV, self).__init__()
        self.weight = TVLoss_weight

    def forward(self, outputs, labels):

        _, _, h, w = outputs.shape

        h_tv = torch.abs(outputs[:, :, 1:, :] - labels[:, :, :h-1, :]).mean()
        w_tv = torch.abs(outputs[:, :, :, 1:] - labels[:, :, :, :w-1]).mean()

        loss = self.weight*(h_tv + w_tv)

        return loss

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1.0):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Itrue = Itrue.reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)
        Ifake = Ifake.reshape(N, C * H * W)

        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)

        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)