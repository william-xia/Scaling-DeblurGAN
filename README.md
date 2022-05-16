# ECE-GY 9143 High Performance Machine Learning Final Project - Scaling Distributed GANs for Image Deblurring
[Paper reference for DeblurGAN implementation](https://arxiv.org/pdf/1711.07064.pdf)

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch

## Train

Train by submitting the job files to HPC via sbatch

```bash
sbatch job_1GPU.s
```

## Other Implementations



