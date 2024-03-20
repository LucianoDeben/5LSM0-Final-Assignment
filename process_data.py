import torch
import torch.nn.functional as F
from torchvision.transforms import (Compose, ConvertImageDtype, Normalize,
                                    PILToTensor, Resize)


def preprocess(img):
        transform = Compose([
        Resize((256, 256)),  
        PILToTensor(), 
        ConvertImageDtype(torch.float32),
        Normalize((0, 0, 0), (1, 1, 1))])  
        return transform(img)

def postprocess(prediction, shape = (256, 256, 18)):
    # Resize prediction to original image size
    prediction = F.interpolate(prediction.unsqueeze(0), size=(shape[0], shape[1]), mode='bilinear', align_corners=False).squeeze(0)
    
    # Apply softmax to get probabilities
    prediction = F.softmax(prediction, dim=0)
    
    # Get the class with the highest probability for each pixel
    prediction = torch.argmax(prediction, dim=0)
    
    # Convert the tensor to a numpy array
    prediction = prediction.cpu().numpy()
    
    return prediction