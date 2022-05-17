# ECE-GY 9143 High Performance Machine Learning Final Project - Scaling Distributed GANs for Image Deblurring
Karan Parikh (kap9580) and William Xia (wx312)

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)
After implementing several different methods to train the deblur GAN, we were able to compare performance metrics between them. 
We show the usage of the trained generative network for deblurring images.

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch
- Dominate

## Train

Download the data from https://drive.google.com/drive/folders/1Ufk3buyGLfY_LbV5iv1ZGE1ZLZGVE8yO?usp=sharing with directory name 'train_dataset' and place it this project's root directory. 
Then, to train the generative network, submit the job files to HPC via sbatch (either 1 or 2 GPU cases).

```bash
sbatch job_1GPU.s
```

## Test

Use the network to deblur sample images using:

```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```

## TO EXTEND WITH MD GAN: 
(NOTE: This is currently on working for 1 node. Future implementations will include working implementation for more nodes)

Download the data from https://drive.google.com/file/d/1CPMBmRj-jBDO2ax4CxkBs9iczIFrs8VA/view?usp=sharing with directory names 'blurred' and 'sharp' place it in the "data" folder. Make sure the  (if data folder is not present, please create it.)

### Prerequisites:
- torch==1.4.0
- torchvision==0.2.2
- tqdm
- tensorboardx
- mpi4py

In order to train over the blurred and sharp images follow these steps:
1. Login in to the HPC and reserve the needed resources using the following sample srun command:
'''bash
srun --nodes=1 —tasks-per-node=1 --cpus-per-task=4 --mem=8GB --time=2:00:00 --gres=gpu:2 --pty /bin/bash
'''
3. Load the MPI module using:
'''bash 
module load openmpi/intel/4.0.5
'''
4. Then, execute "train.py" with the config file "config.json" using the "mpiexec" command:
'''bash
mpiexec -n 4 python train.py —config config.json
'''
For more option with mpiexec refer to the following link: https://www.mpich.org/static/docs/v3.1/www1/mpiexec.html




## References
[DeblurGAN implementation](https://arxiv.org/pdf/1711.07064.pdf)
