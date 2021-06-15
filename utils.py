import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from CelebADataset import CelebADataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
    parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--grad_norm_clip', default=50.0, type=float, help='Clip gradients during training.')
    parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
    parser.add_argument('--trainable', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    args.step = 0  # global step
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


def load_data(root_folder: str) -> DataLoader:
    train_transform = transforms.Compose([
        # transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    train_set = CelebADataset(root_folder, transform=train_transform)

    train_loader = DataLoader(train_set, 16, shuffle=True, num_workers=len(os.sched_getaffinity(0)))

    return train_loader
