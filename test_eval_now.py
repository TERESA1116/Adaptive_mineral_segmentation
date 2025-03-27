import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,classification_report
import seaborn as sns
import pandas as pd

import numpy as np
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
from util import *
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
torch.cuda.is_available()
import shutil
from dataset import *
from model_n import *
from models.FUFH import *

num_classes = 9 

model=SegmentationModel(model_name="DeepLabV3Plus") #DeepLabV3Plus Unet FCN SegNet

loadstateptfile=r'./DeepLabV3Plus/resnet50_imagenet_deeplab.pt'

model.load_state_dict(torch.load(loadstateptfile))
model.to(DEVICE)

loss_fn= DiceLossed(num_classes)


def compute_miou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(np.float32)
        label_cls = (labels == cls).astype(np.float32)

        intersection = np.sum(pred_cls * label_cls)
        union = np.sum(pred_cls + label_cls) - intersection

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)
    return np.nanmean(ious)


def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images1, masks in loader:
            images1=images1.to(DEVICE, dtype=torch.float32)
            masks=masks.to(DEVICE, dtype=torch.float32)
            y_pred,combine_loss = model(images1,masks)
            loss = combine_loss
            epoch_loss += loss.item()

            y_pred_binary = y_pred.argmax(1)
            all_preds.append(y_pred_binary.cpu().numpy())
            all_labels.append(masks.squeeze(1).long().cpu().numpy())

        epoch_loss = epoch_loss / len(loader)

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    all_labels1 = all_labels[all_labels != 1]
    all_preds1 = all_preds[all_labels != 1]
    all_labels2 = all_labels1[all_labels1 != 8]
    all_preds2 = all_preds1[all_labels1 != 8]

    all_labels1 = all_labels2[all_preds2 != 1]
    all_preds1 = all_preds2[all_preds2 != 1]
    all_labels2 = all_labels1[all_preds1 != 8]
    all_preds2 = all_preds1[all_preds1 != 8]

    all_preds = all_preds2
    all_labels = all_labels2

    np.unique(all_preds, return_counts=True)

    # 计算每个类别的混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    print(classification_report(all_labels, all_preds))

    # 计算每个类别的 IoU
    miou = compute_miou(all_preds, all_labels, num_classes)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    sample_num = np.random.randint(0, BATCH_SIZE)
    images1,mask=next(iter(loader))
    images1=images1.to(DEVICE, dtype=torch.float32)
    masks=masks.to(DEVICE, dtype=torch.float32)
    images1=images1[sample_num].unsqueeze(0)
    mask=mask[sample_num]
    logits_mask=model(images1)
    pred_mask=logits_mask.argmax(axis=1)
    img=linear_stretch(images1.cpu().squeeze().numpy()[[2,1,0]], 2)
    img=np.uint8(img*255)


    f, axarr = plt.subplots(1,3) 
    axarr[1].imshow(np.squeeze(mask.numpy()), cmap='jet',vmin=0, vmax=9)
    axarr[0].imshow(np.transpose(img, (1,2,0)))
    axarr[2].imshow(pred_mask.detach().cpu().squeeze(0).numpy(), cmap='jet',vmin=0, vmax=9)
    plt.tight_layout()
    plt.savefig(basedir + '/' + 'test.jpg', pad_inches=0.01, bbox_inches='tight')
    plt.close()

    return epoch_loss, acc, f1, miou, conf_matrix


def save_results_to_csv(acc, f1, miou, conf_matrix, output_dir):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存评估结果到 CSV 文件
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'mIoU'],
        'Value': [acc, f1, miou]
    })

    results_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    # 保存混淆矩阵到 CSV文件
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(conf_matrix.shape[0])],
                                  columns=[f'Class {i}' for i in range(conf_matrix.shape[1])])

    conf_matrix_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_matrix_df.to_csv(conf_matrix_csv_path)

    # 保存混淆矩阵图像
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print(f'Evaluation results saved to {results_csv_path}')
    print(f'Confusion matrix saved to {conf_matrix_csv_path}')

    total_samples = conf_matrix.sum()

    # 将混淆矩阵转换为百分比
    conf_matrix_percent = conf_matrix.astype('float') / total_samples * 100

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(conf_matrix_percent.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix_percent.shape[0])])
    plt.title('Confusion Matrix Percent')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_Percent.png'))
    plt.close()

# 设定统一的输出文件夹路径
output_dir = os.path.join(basedir, 'evaluation_results1')
valid_loss, acc, f1, miou, conf_matrix = evaluate(model, val_loader, loss_fn, DEVICE, num_classes)

print(f"valid_loss: {valid_loss:.4f}")
print(f"acc: {acc:.2f}")
print(f"f1: {f1:.2f}")
print(f"miou: {miou:.2f}")
# 打印混淆矩阵  

print("conf_matrix:")
print(conf_matrix)

save_results_to_csv(acc, f1, miou, conf_matrix, output_dir)

