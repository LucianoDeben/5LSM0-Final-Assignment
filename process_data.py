import torch
import torch.nn.functional as F
from torchvision.transforms import (Compose, ConvertImageDtype, Normalize,
                                    PILToTensor, Resize)


def preprocess(img):
        transform = Compose([
        Resize((256, 256)),
        PILToTensor(),  
        ConvertImageDtype(torch.float32),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(img)
        img = img.unsqueeze(0)
        
        return img

def postprocess(prediction, shape = (256, 256)):
    # Apply softmax to get probabilities
    prediction = F.softmax(prediction, dim=1)
    
    # Get the class with the highest probability for each pixel
    prediction = torch.argmax(prediction, dim=1)
    
    # Convert the tensor to a floating point type
    prediction = prediction.float()
    
    # Add an extra dimension for the number of channels
    prediction = prediction.unsqueeze(0)
    
    # Resize prediction to original image size
    prediction = F.interpolate(prediction, size=shape, mode='bilinear', align_corners=False)
    
    # Remove the extra dimension
    prediction = prediction.squeeze(0)
    
    # Convert the tensor to a numpy array
    prediction = prediction.cpu().detach().numpy()
    
    return prediction