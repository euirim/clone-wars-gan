import torch
import os
import argparse

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_fid import fid_score
from utils import get_dataloader
from config import PARAMS, DEVICE

FID = "fid"


def _write_reals_and_fakes(parent_dir, gen, params, num=1000, device=None):
    dataloader = get_dataloader(params, normalize=False)
    real_dir = f"{parent_dir}/reals"
    fake_dir = f"{parent_dir}/fakes"
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    if not os.path.exists(real_dir):
        os.mkdir(real_dir)
    if not os.path.exists(fake_dir):
        os.mkdir(fake_dir)

    print(f"Writing center-cropped reals to {real_dir}...")
    total = 0
    for data in tqdm(dataloader, total=int(num / params["bsize"])):
        for img in data[0]:
            if total >= num:
                break

            save_image(img, f"{real_dir}/{total}.png")
            total += 1
        if total >= num:
            break

    num = min(num, total)

    print(f"Writing generated fakes to {fake_dir}...")
    for index in tqdm(range(num)):
        z = torch.randn(1, params["nz"], 1, 1, device=device)
        fake = gen(z)[0]
        save_image(fake, f"{fake_dir}/{index}.png")

    return real_dir, fake_dir


def fid(model_path, params, num=1000, device=None):
    gen = torch.load(model_path, map_location=device)["generator"]
    real_dir, fake_dir = _write_reals_and_fakes(
        FID, gen, params, num=num, device=device
    )

    print("Computing FID score...")
    return fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir], batch_size=num, device=device, dims=2048
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved model.")
    parser.add_argument(
        "-n", "--num", help="Maximum number of images to use.", type=int
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to use (defaults to GPU if available, otherwise CPU).",
        type=str,
    )
    args = parser.parse_args()

    num = 1000
    if args.num:
        num = args.num

    device = DEVICE
    if args.device:
        device = torch.device(args.device)

    score = fid(args.model_path, PARAMS, num=num, device=device)
    print(f"FID score: {score}")
