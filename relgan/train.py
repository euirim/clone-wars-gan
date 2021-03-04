"""
Inspired by
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/relativistic_gan/relativistic_gan.py
"""
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter
import random

from utils import get_dataloader
from relgan import Generator, Discriminator

# Set random seed for reproducibility.
seed = 66
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize": 64,  # Batch size during training.
    "img_size": 128,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
    "nz": 100,  # Size of the Z latent vector (the input to the generator).
    "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    "nepochs": 20,  # Number of training epochs.
    "lr": 0.0002,  # Learning rate for optimizers
    "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
    "beta2": 0.999,  # Beta2 hyperparam for Adam optimizer
    "rel_avg_gan": True,  # Use a relativistic average GAN instead of a standard GAN
    "save_epoch": 2,
}  # Save step.

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Configure dataloader
dataloader = get_dataloader(params)

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            sample_batch[0].to(device)[:64], padding=2, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

plt.show()

real_label = 1.0
fake_label = 0.0

# Initialize generator
gen = Generator(params).to(device)

# Initialize discriminator
disc = Discriminator(params).to(device)

# Adversarial loss function
adversarial_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizers
gen_opt = optim.Adam(
    gen.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"])
)
disc_opt = optim.Adam(
    disc.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"])
)

# Tensorboard
writer = SummaryWriter(log_dir="./logs")

# Stores generated images as training progresses.
img_list = []

batches_trained = 0
for epoch in range(params["nepochs"]):
    for i, data in enumerate(tqdm(dataloader), 0):
        real_data = data[0].to(device)

        batch_size = real_data.size(0)

        valid = torch.full((batch_size, 1), real_label, device=device).float()
        fake = torch.full((batch_size, 1), fake_label, device=device).float()

        real_imgs = real_data.float()

        #################
        # Train generator
        #################

        gen_opt.zero_grad()

        # Sample noise
        z = torch.randn(batch_size, params["nz"], device=device)

        # Generate batch of images
        fake_imgs = gen(z)

        real_pred = disc(real_imgs).detach()
        fake_pred = disc(fake_imgs)

        if params["rel_avg_gan"]:
            gen_loss = adversarial_loss(
                fake_pred - real_pred.mean(0, keepdim=True), valid
            )
        else:
            gen_loss = adversarial_loss(fake_pred - real_pred, valid)

        gen_loss = adversarial_loss(disc(fake_imgs), valid)
        gen_loss.backward()
        gen_opt.step()

        #####################
        # Train discriminator
        #####################

        disc_opt.zero_grad()

        real_pred = disc(real_imgs)
        fake_pred = disc(fake_imgs.detach())

        if params["rel_avg_gan"]:
            real_loss = adversarial_loss(
                real_pred - fake_pred.mean(0, keepdim=True), valid
            )
            fake_loss = adversarial_loss(
                fake_pred - real_pred.mean(0, keepdim=True), fake
            )
        else:
            real_loss = adversarial_loss(real_pred - fake_pred, valid)
            fake_loss = adversarial_loss(fake_pred - real_pred, fake)

        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        disc_opt.step()

        # Logging
        if batches_trained % 100 == 0:
            writer.add_scalar("Generator Loss", gen_loss.item(), batches_trained)
            writer.add_scalar("Discriminator Loss", disc_loss.item(), batches_trained)
            writer.add_scalars(
                "Loss",
                {"Generator": gen_loss.item(), "Discriminator": disc_loss.item(),},
                batches_trained,
            )

        batches_trained += 1

    # Save checkpoint
    if epoch % params["save_epoch"] == 0:
        torch.save(
            {
                "generator": gen,
                "discriminator": disc,
                "optimizerG": gen_opt.state_dict(),
                "optimizerD": disc_opt,
                "params": params,
            },
            "model/model_epoch_{}.pth".format(epoch),
        )

    # Logging by epoch (save example output image)
    with torch.no_grad():
        z = torch.randn(params["bsize"], params["nz"], device=device)
        fake_data = gen(z).detach().cpu()
    grid = vutils.make_grid(fake_data, padding=2, normalize=True)
    img_list.append(grid)
    writer.add_image("Outputs", grid, epoch)

# Save final model
torch.save(
    {
        "generator": gen,
        "discriminator": disc,
        "optimizerG": gen_opt.state_dict(),
        "optimizerD": disc_opt,
        "params": params,
    },
    "model/model_final.pth",
)

# Generate animation showing improvements of generator
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save("celeba.gif", dpi=80, writer="imagemagick")
