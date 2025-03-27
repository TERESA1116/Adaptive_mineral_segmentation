import torch
import torch.nn as nn
import torch.nn.functional as F
# from segmentation_models_pytorch import SegNet
from segnet import SegNet
import segmentation_models_pytorch as smp
from config import *

class EdgeDetectionLayer1(nn.Module):
    def __init__(self):
        super(EdgeDetectionLayer1, self).__init__()
        # 使用 Sobel 核进行边缘检测
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # 初始化 Sobel 核
        self.sobel_x.weight.data = torch.tensor([[[[-1, 0, 1],
                                                   [-2, 0, 2],
                                                   [-1, 0, 1]]]], dtype=torch.float32)
        self.sobel_y.weight.data = torch.tensor([[[[-1, -2, -1],
                                                   [0, 0, 0],
                                                   [1, 2, 1]]]], dtype=torch.float32)

        # 将卷积层的权重设为不可学习
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # Convert input to grayscale (if it has 3 channels)
        if x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2)
    
class GatedFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            # nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG
class DSC(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class RES_Dilated_Conv(nn.Module):
    """
    Wide-Focus Residual Block with dilated convolutions.
    """

    def __init__(self, in_channels, out_channels):
        super(RES_Dilated_Conv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=3, bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False)
        
        # Ensure input/output have same number of channels for residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Optional: BatchNorm for better training stability
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # First dilated conv branch
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, p=0.1)

        # Second dilated conv branch
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, p=0.1)

        # Third dilated conv branch
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, p=0.1)

        # Combine the branches
        added = torch.add(x1, x2)
        added = torch.add(added, x3)

        # Apply final convolution
        x_out = self.conv4(added)
        x_out = self.bn4(x_out)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, p=0.1)

        # Add residual connection (shortcut)
        residual = self.shortcut(x)  # Either identity or 1x1 conv
        x_out += residual  # Residual connection
        return x_out
class EdgeDetectionLayer(nn.Module):
    def __init__(self):
        super(EdgeDetectionLayer, self).__init__()
        self.conv_x = nn.Conv2d(3, 3, kernel_size=1, padding=1)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 转换为灰度图像（如果输入是彩色图像）
        # if x.size(1) == 3:
        #     x = x.mean(dim=1, keepdim=True)  # 转为灰度图，简单的通过取均值来实现

        # 使用卷积层提取边缘
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)

        # 计算梯度的幅度（边缘的强度）
        return torch.cat([edge_x,edge_y],dim=1)

class EnhancedDualBranchDeepLabV3Plus(nn.Module):
    def __init__(self, in_channels_xpl, num_classes):
        super(EnhancedDualBranchDeepLabV3Plus, self).__init__()


        self.texture_enhancer = DSC(64,32) 
        
        self.segnet_ppl = smp.DeepLabV3Plus(encoder_name=ENCODER,encoder_weights=WEIGHTS,in_channels=in_channels_xpl,classes=64,activation=None)
        # self.segnet_ppl = SegNet(input_channels=in_channels_xpl, output_channels=64)
        # self.segnet_ppl.load_weights('vgg16_bn-6c64b313.pth')
        self.gatefusion=GatedFusion(32)
        
        # 增强颜色特征的模块
        self.color_enhancer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=5, padding=2)
        )
        # 分支2（xpl）
        # self.segnet_xpl = SegNet(input_channels=in_channels_xpl, output_channels=64)
        self.segnet_xpl  = smp.DeepLabV3Plus(encoder_name=ENCODER,encoder_weights=WEIGHTS,in_channels=in_channels_xpl,classes=64,activation=None)
        # self.segnet_xpl.load_weights('vgg16_bn-6c64b313.pth')
        self.fc = nn.Linear(num_classes * 2, 1)

    def forward(self, x_ppl, x_xpl):
        # 在分支1中进行边缘检测
        
        
        # edges_ppl = self.edge_detector(x_ppl)
        x_ppl = self.segnet_ppl(x_ppl)
        features_ppl = self.texture_enhancer(x_ppl)  # 使用边缘图作为输入
        
        
        
        # 处理分支2
        x_xpl = self.segnet_xpl(x_xpl)
        features_xpl = self.color_enhancer(x_xpl)  # 使用边缘图作为输入


        # 加权融合
        fused_features = self.gatefusion(features_ppl, features_xpl)
        
        # fused_features = self.fusion(features_ppl, features_xpl)

        return fused_features

    def fusion(self, f1, f2):
        
        weight = self.fc(torch.cat((f1.view(f1.size(0), -1), f2.view(f2.size(0), -1)), dim=1))
        return weight * f1 + (1 - weight) * f2
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        

from segmentation_models_pytorch.losses import DiceLoss


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


class FHFU(nn.Module):
    # 定义输入通道和类别数
    # in_channels_ppl = 3  # RGB 图像
    # in_channels_xpl = 3  # RGB 图像
    # num_classes = 2  # 二分类
    
    def __init__(self,inchax1,inchax2,num_classes=9):
        super(FHFU, self).__init__()
        self.num_classes = num_classes
        
        self.inchax1 =inchax1
        # self.inchax2 =inchax2
        # self.DBSEG = EnhancedDualBranchDeepLabV3Plus(inchax1, num_classes)
        # self.segmentation_head = SegmentationHead(
        #     in_channels=32,
        #     out_channels=self.num_classes,
        #     kernel_size=3,
        # )
        self.segnet_ppl = smp.DeepLabV3Plus(encoder_name=ENCODER,encoder_weights=WEIGHTS,in_channels=inchax1,classes=num_classes,activation=None)
        self.dice_loss = DiceLossed(num_classes)
    def forward(self, images, masks=None):  

        # 将输入图像通过模型的主要架构（即Unet）进行前向传播，得到logits。  
        # logits = self.arc(images)  
        logits = self.segnet_ppl(images)  # (B, n_patch, hidden)
        # 如果提供了masks（即标签数据），则计算两种损失。  
        if masks is not  None:  
            # loss1 = DiceLoss(mode='multiclass')(logits, masks.squeeze(1))
            loss1 = self.dice_loss(logits, masks.squeeze(1).long(), softmax=True)
            # 使用交叉熵损失函数计算logits和masks之间的损失
            loss2 = nn.CrossEntropyLoss()(logits, masks.squeeze(1).long())
            loss=loss1+loss2
            # 返回logits和两个损失值
            return logits, loss
        # 如果没有提供masks，则只返回logits。  
        return logits

# # 实例化模型
# model = FHFU(3,3,num_classes=3)

# # 示例输入
# images = torch.randn(1, 6, 256, 256)  # Batch size 1, 256x256 图像


# # 前向传播
# output = model(images)
# print(output.shape)  # 输出形状
