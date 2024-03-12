import torch

def mean_iou(mean_iou, output, target):
    output = torch.argmax(output, dim=1)
    mean_iou.update_state(target.detach().cpu(), output.detach().cpu())
    return mean_iou.result().numpy()
    
def pixel_accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.eq(output, target).float()
    return correct.sum() / correct.numel()

def dice_coefficient(output, target):
    smooth = 1e-6
    output = torch.argmax(output, dim=1).long()  # Convert to long tensor
    target = target.squeeze(1).long()  # Convert to long tensor

    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return dice