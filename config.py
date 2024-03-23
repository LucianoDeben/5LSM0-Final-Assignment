import wandb

# Initialize W&B run
run = wandb.init(project='CityScapes', entity='luciano-deben', config={
    "learning_rate": 0.001,
    "batch_size": 2,
    "num_epochs": 1,
    "num_workers": 8,
    "architecture": "U-Net",
    "dataset": "Cityscapes",
    "optimizer": "Adam",
    "scheduler": "StepLR",
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.1,
    "validation_size": 0.1,
    "weight_decay": 0.0001
})

config = wandb.config