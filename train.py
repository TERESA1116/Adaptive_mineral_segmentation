
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

from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from model_n import *
from models.FUFH import *
from config import *
from dataset import *
from util import *
import logging


name='FCN' #DeepLabV3Plus Unet FCN SegNet FHFU
basedir=rf'./{name}/'
os.makedirs(basedir,exist_ok=True)
outptfile=basedir+f'{ENCODER}_{WEIGHTS}_{name}.pt'

if name=="FHFU":
    model = FHFU(3,3,num_classes=9)
else:
    model=SegmentationModel(model_name=name)  

if loadstate:
    model.load_state_dict(torch.load(loadstateptfile))
model.to(DEVICE)


optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=0.0001)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
outloss={}

trainloss=outloss.setdefault('trainloss',[])
valloss=outloss.setdefault('valloss',[])

best_val_dice_loss=np.Inf
# best_val_bce_loss=np.Inf
best_val_loss=np.Inf

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log", mode='a'),  # 追加模式
        logging.StreamHandler()  # 终端打印
    ]
)


for i in range(EPOCHS):
    outfile=basedir+rf"jpgoutnew/{str(i)}.jpg"
    os.makedirs(os.path.split(outfile)[0],exist_ok=True)
    train_loss, train_acc = train_fn(train_loader,model,optimizer)
    logging.info(f"EPOCHS: {EPOCHS} Training - Avg Loss: {train_loss:.4f}, Avg Accuracy: {train_acc:.4f}")
    valid_loss, val_acc = eval_fn(val_loader,model,outfile)
    logging.info(f"EPOCHS: {EPOCHS} Validation - Avg Loss: {valid_loss:.4f}, Avg Accuracy: {val_acc:.4f}")
    scheduler.step()
    trainloss.append(train_loss)
    valloss.append(valid_loss)
    print(f'Epochs:{i+1}\nTrain_loss --> {train_loss} \nValid_loss --> { valid_loss:} ')

    if valid_loss< best_val_loss:
        torch.save(model.state_dict(),outptfile)
        print('Model Saved')
        best_val_loss= valid_loss
    if i%50==0:
        torch.save(model.state_dict(),outptfile.replace('.pt',f'_{str(i)}.pt'))

import pandas as pd
outcsv=pd.DataFrame(outloss)
outcsv.to_csv(outptfile.replace('.pt','.csv'))

