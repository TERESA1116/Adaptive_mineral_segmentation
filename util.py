import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from config import *
from dataset import *
import numpy as np
import matplotlib.pyplot as plt


def linear_stretch(image, percent):

    # 确保输入是小数并初始化拉伸后的图像
    stretched_image = np.zeros_like(image, dtype=np.float32)
    
    # 对每个波段（通道）分别进行拉伸
    for i in range(image.shape[0]):
        # 获取当前波段
        band = image[i, :, :]
        
        # 将当前波段展平
        flat_band = band.flatten()
        
        # 计算百分比阈值
        low_percent = np.percentile(flat_band, percent)
        high_percent = np.percentile(flat_band, 100 - percent)
        
        # 线性拉伸
        stretched_band = np.clip((band - low_percent) / (high_percent - low_percent), 0, 1)
        
        # 将拉伸后的波段放入结果图像中
        stretched_image[i, :, :] = stretched_band
    
    return stretched_image


def train_fn(data_loader, model, optimizer):
    model.train()
    total_lossn = 0
    num_corrects = 0
    totalsam = 0
    train_bar = tqdm(data_loader)

    for images1, masks in train_bar:
        images1 = images1.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        output, combine_loss = model(images1, masks)
        total_loss = combine_loss
        total_loss.backward()
        optimizer.step()

        total_lossn += total_loss.item()

        # 计算训练精度
        predict = output.argmax(axis=1)
        num_correct = torch.eq(predict, masks.squeeze(1)).sum().float().item()
        num_corrects += num_correct
        totalsam += np.prod(predict.shape)

        accuracy = num_corrects / totalsam
        train_bar.set_description("Train Loss: {:.4f} | Train ACC: {:.4f}".format(total_loss.item(), accuracy))

    avg_loss = total_lossn / len(data_loader)
    avg_acc = num_corrects / totalsam

    return avg_loss, avg_acc

def eval_fn(data_loader, model, outfile):
    model.eval()
    total_lossn = 0
    num_corrects = 0
    totalsam = 0
    test_bar = tqdm(data_loader)

    with torch.no_grad():
        for images1, masks in test_bar:
            images1 = images1.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            logits, combine_loss = model(images1, masks)
            total_loss = combine_loss
            total_lossn += total_loss.item()

            predict = logits.argmax(axis=1)
            num_correct = torch.eq(predict, masks.squeeze(1)).sum().float().item()
            num_corrects += num_correct
            totalsam += np.prod(predict.shape)

            accuracy = num_corrects / totalsam
            test_bar.set_description("Test Loss: {:.4f} | Test ACC: {:.4f}".format(total_loss.item(), accuracy))

        avg_loss = total_lossn / len(data_loader)
        avg_acc = num_corrects / totalsam


        # Visualization
        if outfile is not None:
            sample_num = np.random.randint(0, BATCH_SIZE)
            images1, mask = next(iter(data_loader))
            images1 = images1.to(DEVICE, dtype=torch.float32)
            images1 = images1[sample_num].unsqueeze(0)
            mask = mask[sample_num]
            logits_mask = model(images1)

            pred_mask = logits_mask.argmax(axis=1)
            img = linear_stretch(images1.cpu().squeeze().numpy()[[2, 1, 0]], 2)
            img = np.uint8(img * 255)

            f, axarr = plt.subplots(1, 3)
            axarr[1].imshow(np.squeeze(mask.numpy()), cmap='jet', vmin=0, vmax=9)
            axarr[0].imshow(np.transpose(img, (1, 2, 0)))
            axarr[2].imshow(pred_mask.detach().cpu().squeeze(0).numpy(), cmap='jet', vmin=0, vmax=9)

            plt.tight_layout()
            plt.savefig(outfile, pad_inches=0.01, bbox_inches='tight')
            plt.close()

    return avg_loss, avg_acc
