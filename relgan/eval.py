import torch
import os
import tempfile

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_fid import fid_score
from relgan import Generator, Discriminator
from utils_tmp import get_dataloader


def fid(gen, params, num=50000, device=None):
    dataloader = get_dataloader(params, normalize=False)
    real_dir = tempfile.TemporaryDirectory()
    fake_dir = tempfile.TemporaryDirectory()

    print(f"Writing center-cropped reals to {real_dir.name}...")
    total = 0
    for index, data in enumerate(tqdm(dataloader), 0):
        for img in data[0]:
            if index >= num:
                break

            save_image(img, f"{real_dir.name}/{index}.png")
            total += 1

    num = min(num, total)

    print(f"Writing generated fakes to {fake_dir.name}...")
    fakes = torch.randn(num, params["nz"], device=device)
    for index, fake in enumerate(tqdm(fakes), 0):
        print(fake.size())
        save_image(fake, f"{fake_dir.name}/{index}.png")

    return fid_score.calculate_fid_given_paths(
        [real_dir.name, fake_dir.name], batch_size=num, device=device, dims=num
    )


def test():
    params = {
        "bsize": 128,  # Batch size during training.
        "img_size": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
        "nz": 100,  # Size of the Z latent vector (the input to the generator).
        "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
        "nepochs": 25,  # Number of training epochs.
        "lr": 0.0002,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "save_epoch": 2,
        "num_dataloader_workers": os.cpu_count() // 2,
    }

    MODEL_PATH = "./model/model_epoch_4.pth"
    gen = torch.load(MODEL_PATH)["generator"]

    print(fid(gen, params))
