import torch
import torch.nn as nn
import torch.nn.functional as F




class IOU(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=0):
    
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
                
        return iou

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, smooth=0):
        
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()                            
        dice = (2*intersection + smooth)/(outputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


