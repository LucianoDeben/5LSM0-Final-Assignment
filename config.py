import wandb

# Initialize W&B run
run = wandb.init(
    project="CityScapes",
    entity="luciano-deben",
    config={
        "learning_rate": 0.001,
        "batch_size": 2,
        "num_epochs": 1,
        "num_workers": 8,
        "architecture": "DeepLabV3Plus",
        "dataset": "Cityscapes",
        "optimizer": "Adam",
        "scheduler": "LambdaLR",
        "validation_size": 0.1,
        "weight_decay": 0.0001,
        "grad_accum_steps": 1,
    },
)

config = wandb.config
