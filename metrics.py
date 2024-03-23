import torch


def pixel_accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.eq(output, target).float()
    return correct.sum() / correct.numel()

def dice_coefficient(output, target):
    smooth = 1e-6
    output = torch.argmax(output, dim=1).long() 
    target = target.squeeze(1).long() 

    intersection = (output & (target == output)).float().sum() 
    dice = (2. * intersection + smooth) / (output.numel() + target.numel() + smooth)

    return dice