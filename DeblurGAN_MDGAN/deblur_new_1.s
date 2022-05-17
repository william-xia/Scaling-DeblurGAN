#!/bin/bash
#SBATCH --job-name=deblur_final_mpi
#SBATCH --nodes=3
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
##SBATCH --mail-user=kap9580@nyu.edu 
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2
#SBATCH --output=./deblur_final_mpi.out
#SBATCH --error=./deblur_final_mpi.err

module purge
module load openmpi/intel/4.0.5
mpirun -np 2 python3 ./train.py --config ./config.json
