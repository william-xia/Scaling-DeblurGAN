#!/bin/bash
#SBATCH --job-name=2gpu_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_log_2GPU_100epoch-4.pdf
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wx312@nyu.edu
#SBATCH --gres=gpu:rtx8000:2

python train_dataparallel.py --dataroot training_data --learn_residual --resize_or_crop crop 
--fineSize 256
