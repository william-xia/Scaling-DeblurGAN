# ECE-GY 9143 High Performance Machine Learning Final Project - Scaling Distributed GANs for Image Deblurring
Karan Parikh (kap9580) and William Xia (wx312)

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)
After implementing several different methods to train the deblur GAN, we were able to compare performance metrics between them. 
We show the usage of the trained generative network for deblurring images.

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch

## Train

Train by submitting the job files to HPC via sbatch

```bash
sbatch job_1GPU.s
```

## Test

Use the network to deblur sample images using:

```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```

## References
[DeblurGAN implementation](https://arxiv.org/pdf/1711.07064.pdf)
