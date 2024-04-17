import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from argparse import ArgumentParser

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional.classification import (
    accuracy,
    dice,
    multiclass_jaccard_index,
)
from tqdm import tqdm

import utils
import wandb
from config import config
from model import Model
from process_data import get_data_loader


def get_arg_parser():
    """
    Get the argument parser

    Returns:
    parser: ArgumentParser
    """
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    return parser


def train(train_dataloader, model, criterion, optimizer, device, grad_accum_steps=4):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    train_iou = 0.0

    # Create a GradScaler instance
    scaler = GradScaler()

    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(tqdm(train_dataloader), start=1):
        targets = targets.long().squeeze(dim=1)
        targets = utils.map_id_to_train_id(targets)
        targets[targets == 255] = 19
        inputs, targets = inputs.to(device), targets.to(device)

        # Runs the forward pass with autocasting
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        # Gradient accumulation
        if i % grad_accum_steps == 0 or i == len(train_dataloader):
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # Reset gradients tensors
            optimizer.zero_grad()

        # Update metrics
        train_loss += loss.item()
        outputs_max = torch.argmax(outputs, dim=1)
        train_acc += accuracy(
            outputs_max, targets, task="multiclass", num_classes=20
        ).detach()
        train_dice += dice(outputs, targets, ignore_index=19).detach()
        train_iou += multiclass_jaccard_index(
            outputs, targets, num_classes=20, ignore_index=19
        ).detach()

    num_batches = len(train_dataloader)
    return (
        (train_loss / num_batches),
        (train_acc / num_batches),
        (train_dice / num_batches),
        (train_iou / num_batches),
    )


def validate(val_dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    val_iou = 0.0

    # Create a GradScaler instance
    scaler = GradScaler()

    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader):
            targets = targets.long().squeeze(dim=1)
            targets = utils.map_id_to_train_id(targets)
            targets[targets == 255] = 19
            inputs, targets = inputs.to(device), targets.to(device)

            # Use autocast to enable mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale the loss using GradScaler
            scaler.scale(loss)

            val_loss += loss.detach()
            outputs_max = torch.argmax(outputs, dim=1)
            val_acc += accuracy(
                outputs_max, targets, task="multiclass", num_classes=20
            ).detach()
            val_dice += dice(outputs, targets, ignore_index=19).detach()
            val_iou += multiclass_jaccard_index(
                outputs, targets, num_classes=20, ignore_index=19
            ).detach()

    num_batches = len(val_dataloader)
    return (
        (val_loss / num_batches).item(),
        (val_acc / num_batches).item(),
        (val_dice / num_batches).item(),
        (val_iou / num_batches).item(),
    )


def main(args):
    """
    Main function for training the model

    Args:
    args: Namespace

    Returns:
    None
    """

    # Data loading
    train_loader, val_loader = get_data_loader(
        args, config.batch_size, config.num_workers, config.validation_size
    )

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move your model to the device
    model = Model()

    # Freeze the backbone parameters
    # for name, param in model.named_parameters():
    #     if "backbone" in name:
    #         param.requires_grad = False

    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training.")
        model = nn.parallel.DistributedDataParallel(model)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=19)

    # Define the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
    )

    # Define the function for adjusting learning rate at each epoch
    lr_lambda = lambda epoch: (1 - epoch / config.num_epochs) ** 0.9

    # Create the scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc, train_dice, train_iou = train(
            train_loader, model, criterion, optimizer, device, config.grad_accum_steps
        )
        val_loss, val_acc, val_dice, val_iou = validate(
            val_loader, model, criterion, device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        wandb.log(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )
        scheduler.step()

        # Conditionally Save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./model.pth")


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
