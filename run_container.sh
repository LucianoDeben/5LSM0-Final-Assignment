export WANDB_API_KEY=61c2385b12425146821e27728eabb6fc3238556d
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

# Install Keras
pip install keras

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
