import medmnist
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO


def load_medmnist(data_flag="organamnist", batch_size=128, image_size=28, as_rgb=False):
    """Returns (train_loader, val_loader, test_loader, info)."""
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    steps = []
    if image_size != 28:
        steps.append(transforms.Resize((image_size, image_size),
                                       interpolation=transforms.InterpolationMode.BILINEAR))
    n_channels = 3 if as_rgb else info["n_channels"]
    steps += [transforms.ToTensor(),
              transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels)]
    transform = transforms.Compose(steps)

    splits = {
        split: DataClass(split=split, transform=transform, download=True, as_rgb=as_rgb)
        for split in ("train", "val", "test")
    }

    loaders = tuple(
        data.DataLoader(splits[split], batch_size=batch_size, shuffle=(split == "train"))
        for split in ("train", "val", "test")
    )
    return (*loaders, info)
