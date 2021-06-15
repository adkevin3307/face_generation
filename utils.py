import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from CelebADataset import CelebADataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_norm_clip', default=50.0, type=float)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--trainable', action='store_true')
    args = parser.parse_args()

    print('=' * 100)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 100)

    return args


def load_data(root_folder: str) -> DataLoader:
    train_transform = transforms.Compose([
        # transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CelebADataset(root_folder, transform=train_transform)

    train_loader = DataLoader(train_set, 16, shuffle=True, num_workers=len(os.sched_getaffinity(0)))

    return train_loader
