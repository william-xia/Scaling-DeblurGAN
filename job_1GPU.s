#!/bin/bash
#SBATCH --job-name=1GPU_100epoch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_log_1GPU_100epoch-5.pdf
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wx312@nyu.edu
#SBATCH --gres=gpu:rtx8000:1

python train.py --dataroot train_dataset --learn_residual --resize_or_crop crop --fineSize 256
