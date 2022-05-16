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

## References
[DeblurGAN implementation](https://arxiv.org/pdf/1711.07064.pdf)
