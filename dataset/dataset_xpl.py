import os
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


class CustomDataset(Dataset):
    def __init__(self, xpl_dir, ppl_dir, mode='train', transform=None):
        self.xpl_dir = xpl_dir
        self.ppl_dir = ppl_dir
        self.transform = transform
        self.mode = mode
        
        self.xplimage_names = os.listdir(xpl_dir+'/'+mode)
        self.xplimage_namesexits=[]
        # print(self.xplimage_names)
        for xplimage_name in self.xplimage_names:
            labelnames=xplimage_name.split('_')
            labelname='_'.join(labelnames[:-1]+['color','mask']+labelnames[-1:])
            label_path = os.path.join(self.xpl_dir,self.mode+'annot', labelname.replace('jpg','png'))
            xpl_path = os.path.join(self.xpl_dir, self.mode,xplimage_name)
            ppl_path = os.path.join(self.ppl_dir, self.mode,xplimage_name.replace('xpl','ppl'))
            # print(ppl_path,os.path.exists(ppl_path) )
            # print(os.path.exists(ppl_path) , os.path.exists(xpl_path) , os.path.exists(label_path))
            if  os.path.exists(ppl_path) and os.path.exists(xpl_path) and os.path.exists(label_path):
                self.xplimage_namesexits.append(xplimage_name)

    def __len__(self):
        return len(self.xplimage_namesexits)

    def __getitem__(self, idx):
        xplimage_name = self.xplimage_namesexits[idx]
        labelnames=xplimage_name.split('_')
        labelname='_'.join(labelnames[:-1]+['color','mask']+labelnames[-1:])
        # Load optical image (4 channels)
        xpl_path = os.path.join(self.xpl_dir, self.mode,xplimage_name)
        xpl_image = Image.open(xpl_path)
        
        
        # Load radar image (1 channel)
        ppl_path = os.path.join(self.ppl_dir, self.mode,xplimage_name.replace('xpl','ppl'))
        ppl_image = Image.open(ppl_path)
        
        
        # Load label image (binary mask)
        label_path = os.path.join(self.xpl_dir,self.mode+'annot', labelname.replace('jpg','png'))
        
        label_image = np.array(Image.open(label_path).convert("L") )

        # Stack optical and radar images
        xpl_imagear = np.array(xpl_image)
        ppl_imagear = np.array(ppl_image)
        
        try:
            h,w,c=xpl_imagear.shape
            
            h=h-2
            w=w-2
            xpl_imagear=(xpl_imagear[:h,:w,:])  # Shape (H, W, 5)
            ppl_imagear=ppl_imagear[:h,:w,:]
            label_image=label_image[:h,:w]
        except:
            h,w,c=ppl_imagear.shape
            h=h-2
            w=w-2
            xpl_imagear=(xpl_imagear[:h,:w,:])  # Shape (H, W, 5)
            ppl_imagear=ppl_imagear[:h,:w,:]
            label_image=label_image[:h,:w]
        if self.transform:
            augmented = self.transform(image=xpl_imagear, mask=label_image)
            xpl_imagear = augmented['image']
            label_image = augmented['mask']
        # label_image = np.transpose(label_image, (2, 0, 1))
        # print(label_image.shape)
        label_image[label_image == 15] = 1
        label_image[label_image == 38] = 2
        label_image[label_image == 53] = 3
        label_image[label_image == 75] = 4
        label_image[label_image == 90] = 5
        label_image[label_image == 113] = 6
        label_image[label_image == 128] = 7
        label_image[label_image == 255] = 8
        
        return xpl_imagear, label_image


# Example usage
xpl_dir = r'data/xpl'
ppl_dir = r'data/ppl'

transform = A.Compose([
    A.Resize(256, 256),
    A.ToFloat(255),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    # A.RandomBrightnessContrast(),
    ToTensorV2(),
])

test_dataset = CustomDataset(xpl_dir, ppl_dir, mode='test', transform=transform)
list(test_dataset)
# Create dataset
train_dataset = CustomDataset(xpl_dir, ppl_dir, mode='train', transform=transform)
# list(train_dataset)


# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # # 示例遍历训练数据
# for data in train_loader:
#     image1,mask= data
#     print(image1.shape)
#     print(np.unique(mask),mask.shape)  # 打印掩码中的类别标签
#     break
