# ECE-GY 9143 High Performance Machine Learning Final Project - Scaling Distributed GANs for Image Deblurring
Karan Parikh (kap9580) and William Xia (wx312)

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)
After implementing several different methods to train the deblur GAN, we were able to compare performance metrics between them. 
We show the usage of the trained generative network for deblurring images.

![alt text](https://github.com/KupynOrest/DeblurGAN/blob/master/images/animation3.gif)
![alt text](https://github.com/KupynOrest/DeblurGAN/blob/master/images/animation4.gif)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch
- Dominate (install using 'pip install dominate')

## Train

Download the data from https://drive.google.com/drive/folders/1Ufk3buyGLfY_LbV5iv1ZGE1ZLZGVE8yO?usp=sharing with directory name 'train_dataset' and place it this project's root directory. 
To train using a locally, simply run the following command in the project directory:

```bash
python train.py --dataroot training_data --learn_residual --resize_or_crop crop --fineSize 256
```

To train the generative network on NYU HPC, first create the following sbatch file with a request for 1, 2 or 4 GPUS.

```bash
#!/bin/bash
#SBATCH --job-name=1GPU_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=results.txt
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1

python train_dataparallel.py --dataroot training_data --learn_residual --resize_or_crop crop --fineSize 256
```

Then submit it using the following command:

```bash
sbatch job_1GPU.s
```

## Test

Use the network to deblur sample images using:

```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```

## TO SCALE WITH MD GAN: 

We address the problem of distributing Deblur GAN so that they are able to train over datasets that are spread on multiple workers. MD-GAN exhibits a reduction by a factor of two of the learning complexity on each worker node, while providing better performances than federated learning on both datasets. \
(NOTE: This is currently only working for 1 node. Future implementations will include working implementation for more nodes) \
(You can use this code by removing the MPI references in the code and run it as regular DeblurGAN)

![alt text](https://github.com/william-xia/Scaling-DeblurGAN/blob/main/git%20images/Screen%20Shot%202022-05-17%20at%2012.16.26%20AM.png?raw=true)
## How to run
### Prerequisites:
- torch==1.4.0
- torchvision==0.2.2
- tqdm
- tensorboardx
- mpi4py

## Train

Download the data from https://drive.google.com/file/d/1CPMBmRj-jBDO2ax4CxkBs9iczIFrs8VA/view?usp=sharing with directory names 'blurred' and 'sharp' place it in the "data" folder. Make sure the  (if data folder is not present, please create it.)

In order to train over the blurred and sharp images, create this batch file and run it using sbatch:
``` bash
#!/bin/bash
#SBATCH --job-name=deblur_final_mpi
#SBATCH --nodes=3
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2
#SBATCH --output=./deblur_final_mpi.out
#SBATCH --error=./deblur_final_mpi.err

module purge
module load openmpi/intel/4.0.5
mpirun -np 2 python3 ./train.py --config ./config.json
```

## Test
In order to deblur images run "deblur_image.py" with the following command:

```bash
python deblur_image.py --blurred <blurred image directory> --deblurred <output image directory> --resume <trained weights directory>
```

## References
[DeblurGAN implementation](https://arxiv.org/pdf/1711.07064.pdf) \
[MDGAN implementation](https://arxiv.org/pdf/1811.03850.pdf) \
The images were taken from GoPRO test dataset - [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release)
