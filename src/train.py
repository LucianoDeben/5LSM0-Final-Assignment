"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import Model
from metrics import pixel_accuracy, dice_coefficient, mean_iou
import torch
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras

DATA_PATH = "./data/Cityscapes"
BATCH_SIZE = 2
NUM_WORKERS = 1

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

def get_data_loader(args, batch_size, num_workers):
    # Define the transform for the images
    transform = v2.Compose([
        v2.Resize((256, 256)),  
        v2.PILToTensor(), 
        v2.ToDtype(torch.float32),
        v2.Normalize((0, 0, 0), (1, 1, 1))
    ])
    # Define the transform for the labels
    target_transform = v2.Compose([
        v2.Resize((256, 256), interpolation=0),
        v2.PILToTensor(), 
        v2.ToDtype(torch.float32)
    ])

    # Load the dataset
    dataset = Cityscapes(DATA_PATH, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
    
    # Define the size of the validation set
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Define the data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def train(train_dataloader, model, criterion, optimizer, device, m_iou):
    model.train()
    train_loss = 0
    train_acc = 0
    train_dice = 0
    train_m_iou = 0
    for inputs, targets in tqdm(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze(1).long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += pixel_accuracy(outputs, targets).item()
        train_dice += dice_coefficient(outputs, targets).item()
        train_m_iou += float(mean_iou(mean_iou=m_iou, output=outputs, target=targets))

    return train_loss / len(train_dataloader), train_acc / len(train_dataloader), train_dice / len(train_dataloader), train_m_iou / len(train_dataloader)

def validate(val_dataloader, model, criterion, device, m_iou):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_dice = 0
    val_m_iou = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())

            val_loss += loss.item()
            val_acc += pixel_accuracy(outputs, targets).item()
            val_dice += dice_coefficient(outputs, targets).item()
            val_m_iou += float(mean_iou(mean_iou=m_iou, output=outputs, target=targets))
    return val_loss / len(val_dataloader), val_acc / len(val_dataloader), val_dice / len(val_dataloader), m_iou.result()


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    train_loader, val_loader = get_data_loader(args, BATCH_SIZE, NUM_WORKERS)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model = Model().to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Define the number of epochs
    n_epochs = 1
    
    # Define the mean IOU metric
    m_iou_train = keras.metrics.MeanIoU(num_classes=34)
    m_iou_val = keras.metrics.MeanIoU(num_classes=34)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_dices = []
    val_dices = []
    train_mious = []
    val_mious = [] 

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc, train_dice, train_miou = train(train_loader, model, criterion, optimizer, device, m_iou_train)
        val_loss, val_acc, val_dice, val_miou = validate(val_loader, model, criterion, device, m_iou_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_mious.append(train_miou)
        val_mious.append(val_miou)
        
        print(f"Train Loss: {train_loss:.4f}, Train IOU: {train_miou:.4f}, Train Acc: {train_acc:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IOU: {val_miou}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}")
        scheduler.step()
        
    # Plot the training and validation metrics
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)   
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 4, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 4, 3)
    plt.plot(train_dices, label='Training Dice')
    plt.plot(val_dices, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.subplot(1, 4, 4)
    plt.plot(train_mious, label='Training IOU')
    plt.plot(val_mious, label='Validation IOU')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.legend()
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), "./models/model.pth")

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
