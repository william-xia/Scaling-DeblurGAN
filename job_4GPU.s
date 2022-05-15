#!/bin/bash
#SBATCH --job-name=4gpu_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_log_4GPU_100epoch-2.pdf
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wx312@nyu.edu
#SBATCH --gres=gpu:rtx8000:4

python train_dataparallel.py --dataroot training_data --learn_residual --resize_or_crop crop 
--fineSize 256
