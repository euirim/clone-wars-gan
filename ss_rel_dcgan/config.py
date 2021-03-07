import os
import random
import torch

# Parameters to define the model.
PARAMS = {
    "bsize": 64,  # Batch size during training.
    "img_size": 128,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
    "nz": 100,  # Size of the Z latent vector (the input to the generator).
    "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    "nepochs": 30,  # Number of training epochs.
    "lr": 0.0002,  # Learning rate for optimizers
    "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
    "beta2": 0.999,  # Beta2 hyperparam for Adam optimizer
    "rel_avg_gan": True,  # Use a relativistic average GAN instead of a standard GAN
    "save_epoch": 2,
    "num_dataloader_workers": os.cpu_count() // 2,
    "nrots": 4,
    "gp_weight": 0.2,  # Loss term weight for gradient penalty
    "rot_weight_d": 1.0,  # Loss term weight for rotations for discriminator
    "rot_weight_g": 0.5,  # Loss term weight for rotations for generator
}  # Save step.

# Set random seed for reproducibility.
SEED = 66
random.seed(SEED)
torch.manual_seed(SEED)

# Use GPU is available else use CPU.
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
