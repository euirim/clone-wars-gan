import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = "../data/full"


def get_dataloader(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose(
        [
            transforms.Resize(params["img_size"]),
            transforms.CenterCrop(params["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["bsize"], shuffle=True
    )

    return dataloader
