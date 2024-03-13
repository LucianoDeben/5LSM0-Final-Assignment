import torch
import torch.nn.functional as F
import torchvision.transforms as v2


def preprocess(img):
        transform = v2.Compose([
        v2.Resize((256, 256)),  
        v2.PILToTensor(), 
        v2.ToDtype(torch.float32),
        v2.Normalize((0, 0, 0), (1, 1, 1))])  
        return transform(img)

def postprocess(prediction, shape = (1024, 2048, 34)):
    # Resize prediction to original image size
    prediction = F.interpolate(prediction.unsqueeze(0), size=shape, mode='bilinear', align_corners=False).squeeze(0)
    
    # Apply softmax to get probabilities
    prediction = F.softmax(prediction, dim=0)
    
    # Get the class with the highest probability for each pixel
    prediction = torch.argmax(prediction, dim=0)
    
    # Convert the tensor to a numpy array
    prediction = prediction.cpu().numpy()
    
    return prediction