import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification import (accuracy, dice,
                                                    multiclass_jaccard_index)
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

import utils
import wandb
from config import config
from model import Model


def get_arg_parser():
    """
    Get the argument parser
    
    Returns:
    parser: ArgumentParser
    """
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    return parser

def get_data_loader(args, batch_size, num_workers, spatial_dims):
    """
    
    Get the data loader
    
    Args:
    args: Namespace
    batch_size: int
    num_workers: int
    
    Returns:
    train_loader: DataLoader
    val_loader: DataLoader
    """
    transform = v2.Compose([
            v2.Resize(spatial_dims),  
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    target_transform = v2.Compose([
        v2.Resize(spatial_dims, interpolation=InterpolationMode.NEAREST), 
        v2.ToImage()
        ])

    # Load the dataset
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
    
    # Define the size of the validation set
    val_size = int(config.validation_size * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Define the data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def train(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    train_iou = 0.0
    
    for inputs, targets in tqdm(train_dataloader):
        print(targets)
        targets = targets.long().squeeze(dim = 1)   
        targets = utils.map_id_to_train_id(targets)
        targets[targets == 255] = 19
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach()
        outputs_max = torch.argmax(outputs, dim=1)
        #outputs_max = torch.unsqueeze(outputs_max, 1)
        
        train_acc += accuracy(outputs_max, targets, task="multiclass", num_classes=20, ignore_index=19).detach()
        train_dice += dice(outputs, targets, ignore_index=19).detach()
        train_iou += multiclass_jaccard_index(outputs, targets, num_classes=20, ignore_index=19).detach()

    num_batches = len(train_dataloader)
    return (train_loss / num_batches).item(), (train_acc / num_batches).item(), (train_dice / num_batches).item(), (train_iou / num_batches).item()

def validate(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader):
            print(targets)
            targets = targets.long().squeeze(dim = 1)   
            targets = utils.map_id_to_train_id(targets)
            targets[targets == 255] = 19
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            val_loss += loss.detach()
            outputs_max = torch.argmax(outputs, dim=1)
            #outputs_max = torch.unsqueeze(outputs_max, 1)
            
            val_acc += accuracy(outputs_max, targets, task="multiclass", num_classes=20).detach()
            val_dice += dice(outputs, targets, ignore_index=19).detach()
            val_iou += multiclass_jaccard_index(outputs, targets, num_classes=20, ignore_index=19).detach()

    num_batches = len(val_dataloader)
    return (val_loss / num_batches).item(), (val_acc / num_batches).item(), (val_dice / num_batches).item(), (val_iou / num_batches).item()	


def main(args):
    """
    Main function for training the model
    
    Args:
    args: Namespace
    
    Returns:
    None
    """

    # Data loading
    train_loader, val_loader = get_data_loader(args, config.batch_size, config.num_workers, spatial_dims=(256, 256))
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model = Model().to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=19)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # Define the number of epochs
    n_epochs = config.num_epochs
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []
    
    wandb.watch(model, log_freq=100)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc, train_dice, train_iou = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_dice, val_iou = validate(val_loader, model, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        wandb.log({
            "epoch": epoch,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "train_dice": train_dice,
    "train_lr": optimizer.param_groups[0]["lr"],
    })
        wandb.log({
            "epoch": epoch,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "val_dice": val_dice,
    "val_lr": optimizer.param_groups[0]["lr"]
    })
        scheduler.step()
        
    # Save model
    torch.save(model.state_dict(), "./model.pth")

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
