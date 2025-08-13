#!/bin/bash
#SBATCH --job-name=qwen2vl_lora
#SBATCH --output=logs/qwen2vl_lora_%j.out
#SBATCH --error=logs/qwen2vl_lora_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --signal=SIGUSR1@90

# Load environment
module load anaconda
source ~/.bashrc
conda activate qwen2vl

# Optional: ensure logs directory exists
mkdir -p logs

# Run training
srun python3 train_qwen_lora_lightning.py

# # Optionally, run a configurable version
# srun python3 train_qwen2_configurable.py \
#   --epochs 5 \
#   --lr 5e-5