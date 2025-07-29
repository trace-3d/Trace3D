#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def mask_mse(network_output, gt, mask=None):
    mask = mask.float()
    loss =((network_output - gt) ** 2).sum(-1)
    loss=loss*mask
    loss = loss.sum() / mask.sum()
    return loss

def mask_cross_entropy(network_output, gt, mask=None):
    loss_func=torch.nn.functional.cross_entropy
    mask = mask.float()
    loss= loss_func(network_output,gt)
    loss=loss*mask
    loss = loss.sum() / mask.sum()
    return loss

#Copyright Contrastive-Lift (NeurIPS 2023 Spotlight)
#https://github.com/yashbhalgat/Contrastive-Lift
def contrastive_loss(features, instance_labels, temperature):
    bsize = features.size(0)
    masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone())
    masks = masks.fill_diagonal_(0, wrap=False)

    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1)
    # temperature = 1 for positive pairs and temperature for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature
    temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))

    similarity_kernel = torch.exp(-distance_sq/temperature)
    logits = torch.exp(similarity_kernel)

    p = torch.mul(logits, masks).sum(dim=-1)
    Z = logits.sum(dim=-1)

    prob = torch.div(p, Z)
    prob_masked = torch.masked_select(prob, prob.ne(0))
    loss = -prob_masked.log().sum()/bsize
    return loss

def mask_mse1(network_output, gt, mask=None):
    mask=mask.float()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    network_output =  network_output*mask
    gt = gt*mask
    loss=loss_fn(network_output, gt)
    loss*=len(network_output/mask.sum())
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

