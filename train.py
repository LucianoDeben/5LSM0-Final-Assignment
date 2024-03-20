"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Cityscapes
from torchvision.transforms import v2
from tqdm import tqdm

import utils
import wandb
from config import config
from metrics import dice_coefficient, pixel_accuracy
from model import Model


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
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
    
    # Define the size of the validation set
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Define the data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def train(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_dice = 0
    for inputs, targets in tqdm(train_dataloader):
        targets = utils.map_id_to_train_id(targets)
        print(targets.unique())
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze(1).long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += pixel_accuracy(outputs, targets).item()
        train_dice += dice_coefficient(outputs, targets).item()

    return train_loss / len(train_dataloader), train_acc / len(train_dataloader), train_dice / len(train_dataloader)

def validate(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_dice = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())

            val_loss += loss.item()
            val_acc += pixel_accuracy(outputs, targets).item()
            val_dice += dice_coefficient(outputs, targets).item()
    return val_loss / len(val_dataloader), val_acc / len(val_dataloader), val_dice / len(val_dataloader)


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    train_loader, val_loader = get_data_loader(args, config.batch_size, config.num_workers)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model = Model().to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

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
    
    wandb.watch(model, log_freq=100)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc, train_dice = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_dice = validate(val_loader, model, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        wandb.log({
            "epoch": epoch,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "train_dice": train_dice
    })
        wandb.log({
            "epoch": epoch,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "val_dice": val_dice
    })
        scheduler.step()
        
    # Save model
    torch.save(model.state_dict(), "./model.pth")

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
