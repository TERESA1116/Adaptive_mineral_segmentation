import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from config import *
import numpy as np
import torch
import torchvision.models.segmentation
from models.segnet import *
from models.FCN import *

class DiceLossed(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossed, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def total_loss(logits,masks,dice_loss):
    masks = torch.squeeze(masks, dim = 1).long()
    loss2 = nn.CrossEntropyLoss()(logits, masks)/masks.size(0)
    loss1 = dice_loss(logits, masks.squeeze(1).long(), softmax=True)

    return loss2+loss1


class SegmentationModel(nn.Module):
    def __init__(self,model_name='DeepLabV3Plus'):
        super(SegmentationModel, self).__init__()
        
        # Define the UNet model

        num_classes = 9  # 根据数据集的类数

        if model_name=="DeepLabV3Plus":
            self.arc = smp.DeepLabV3Plus(encoder_name=ENCODER,encoder_weights=WEIGHTS,in_channels=6,classes=num_classes,activation=None)
        if model_name=="Unet":
            self.arc = smp.Unet(encoder_name=ENCODER,encoder_weights=WEIGHTS,in_channels=6,classes=num_classes,activation=None)
        elif model_name=="FCN":
            vgg_model = VGGNet(requires_grad=False)
            self.arc  = FCN32s(pretrained_net=vgg_model, n_class=num_classes)
        elif model_name=="SegNet":
            self.arc  = SegNet(6, num_classes)
        
        self.dice_loss = DiceLossed(9)
        
    def forward(self, data, masks=None):
        logits = self.arc(data)
        if masks is not None:
            masks = torch.squeeze(masks, dim = 1).long()
            loss2 = nn.CrossEntropyLoss()(logits, masks)/masks.size(0)

            loss1 = self.dice_loss(logits, masks.squeeze(1), softmax=True)
            loss=loss1+loss2
            return logits, loss
        
        return logits
