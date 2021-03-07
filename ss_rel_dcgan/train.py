from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.tensorboard import SummaryWriter

from config import PARAMS as params
from config import DEVICE as device

from utils import get_dataloader
from ss_rel_dcgan import Generator, Discriminator


# Custom weight initialization
# From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


print(device, " used.")

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
            sample_batch[0].to(device)[:64], padding=2, normalize=True,
        ).cpu(),
        (1, 2, 0),
    )
)

plt.show()

real_label = 1.0
fake_label = 0.0

# Initialize generator
gen = Generator(params).to(device)
# Apply custom weight initialization
gen.apply(weights_init)

# Initialize discriminator
disc = Discriminator(params).to(device)
# Apply custom weight initialization
disc.apply(weights_init)

# Binary Cross Entropy Loss
criterion = nn.BCEWithLogitsLoss().to(device)

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

# Fixed noise for visualization of generator progression
fixed_visualization_noise = torch.randn(64, params["nz"], 1, 1, device=device)

batches_trained = 0
for epoch in range(1, params["nepochs"] + 1):
    print(f"* Epoch {epoch}")
    for i, data in enumerate(tqdm(dataloader), 0):
        # Get data and transfer to GPU/CPU
        real_imgs = data[0].to(device)
        # Get batch size (can be different than params for last batch)
        batch_size = real_imgs.size(0)

        #################
        # Train Discriminator
        #################

        disc.zero_grad()
        # Create labels for the real data. (label=1)
        real_labels = torch.full((batch_size * 4,), real_label, device=device).float()
        fake_labels = torch.full((batch_size * 4,), fake_label, device=device).float()

        # Rotate all of the real images by 0, 90, 180, and 270 degrees
        x = real_imgs
        x_90 = x.transpose(2, 3)
        x_180 = x.flip(2, 3)
        x_270 = x.transpose(2, 3).flip(2, 3)
        real_imgs = torch.cat((x, x_90, x_180, x_270), 0)

        # Sample random data from a unit normal distribution.
        noise = torch.randn(batch_size, params["nz"], 1, 1, device=device)
        # Generate fake data (images).
        fake_imgs = gen(noise)

        # Rotate all of the fake images by 0, 90, 180, and 270 degrees
        x = fake_imgs
        x_90 = x.transpose(2, 3)
        x_180 = x.flip(2, 3)
        x_270 = x.transpose(2, 3).flip(2, 3)
        fake_imgs = torch.cat((x, x_90, x_180, x_270), 0)

        y_real_logits, y_real_rots_logits = disc(real_imgs)
        y_real_logits = y_real_logits.view(-1)
        y_real_rots_logits = y_real_rots_logits.view(batch_size * 4, -1)

        # Compute the one-hot rotational labels
        rot_labels = torch.zeros(4 * batch_size, device=device)
        for i in range(4 * batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2 * batch_size:
                rot_labels[i] = 1
            elif i < 3 * batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3
        rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()

        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        y_fake_logits, y_fake_rots_logits = disc(fake_imgs.detach())
        y_fake_logits = y_fake_logits.view(-1)
        y_fake_rots_logits = y_fake_rots_logits.view(-1)

        # Net discriminator relativistic loss.
        disc_loss = (
            criterion(y_real_logits - torch.mean(y_fake_logits), real_labels)
            + criterion(y_fake_logits - torch.mean(y_real_logits), fake_labels)
        ) / 2

        # Compute gradient penalty and add this to the discriminator loss
        # alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
        # alpha = alpha.expand_as(real_imgs)
        # interpolated = Variable(alpha * real_imgs + (1 - alpha) * fake_imgs, requires_grad=True, device=device)
        # interpolated_logits, _ = disc(interpolated)
        # interpolated_probs = nn.Sigmoid(interpolated_logits)
        # gradients = torch_grad(outputs=interpolated_probs, inputs=interpolated,
        #                        grad_outputs=torch.ones(interpolated_probs.size(), device=device),
        #                        create_graph=True, retain_graph=True)[0]
        # gradients = gradients.view(real_imgs.size(0), -1)

        # # TODO add these to tensorboard
        # gradients.norm(2, dim=1).sum().data

        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # disc_loss += params["gp_weight"] * ((gradients_norm - 1) ** 2).mean()

        # Loss term for rotations
        disc_loss += params["rot_weight_d"] * torch.sum(
            F.binary_cross_entropy_with_logits(
                input=y_real_rots_logits, target=rot_labels
            )
        )

        # Backprop
        disc_loss.backward()
        # Update discriminator parameters.
        disc_opt.step()

        #################
        # Train generator
        #################

        # Make accumulated gradients of the generator zero.
        gen.zero_grad()

        y_real_logits, y_real_rots_logits = disc(
            real_imgs
        )  # necessary to avoid gradient problems
        y_real_logits = y_real_logits.view(-1)
        y_real_rots_logits = y_real_rots_logits.view(batch_size * 4, -1)
        y_fake_logits, y_fake_rots_logits = disc(fake_imgs)
        y_fake_logits = y_fake_logits.view(-1)
        y_fake_rots_logits = y_fake_rots_logits.view(batch_size * 4, -1)

        gen_loss = (
            criterion(y_real_logits - torch.mean(y_fake_logits), fake_labels)
            + criterion(y_fake_logits - torch.mean(y_real_logits), real_labels)
        ) / 2
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()

        gen_loss += params["rot_weight_g"] * torch.sum(
            F.binary_cross_entropy_with_logits(
                input=y_fake_rots_logits, target=rot_labels
            )
        )

        gen_loss.backward()
        gen_opt.step()

        # Logging
        if batches_trained % 100 == 0:
            writer.add_scalar("Generator Loss", gen_loss.item(), batches_trained)
            writer.add_scalar("Discriminator Loss", disc_loss.item(), batches_trained)
            writer.add_scalars(
                "Loss",
                {"Generator": gen_loss.item(), "Discriminator": disc_loss.item(),},
                batches_trained,
            )
            with torch.no_grad():
                fake_data = gen(fixed_visualization_noise).detach().cpu()
            grid = vutils.make_grid(fake_data, padding=2, normalize=True)
            writer.add_image("Outputs", grid, batches_trained)

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
        fake_data = gen(fixed_visualization_noise).detach().cpu()
    grid = vutils.make_grid(fake_data, padding=2, normalize=True)
    img_list.append(grid)
    vutils.save_image(grid, f"outputs/epoch_{epoch}.png")

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
anim.save("celeba.gif", dpi=300, writer="imagemagick")
