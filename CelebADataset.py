import os
from typing import Callable
from PIL import Image
import torch
from torch.utils.data import Dataset


def get_CelebA_data(root_folder: str) -> tuple:
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []

    with open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r') as txt_file:
        num_imgs = int(txt_file.readline()[:-1])

        _ = txt_file.readline()[:-1].split(' ')

        for _ in range(num_imgs):
            line = txt_file.readline()[:-1].split(' ')
            label = list(map(int, line[2:]))

            label_list.append(label)

    return (img_list, label_list)


class CelebADataset(Dataset):
    def __init__(self, root_folder: str, transform: Callable = None) -> None:
        self.root_folder = root_folder
        self.transform = transform

        assert os.path.isdir(self.root_folder), f'{self.root_folder} is not a valid directory'

        self.image_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40

        print("> Found %d images..." % (len(self.image_list)))

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> tuple:
        image_folder = os.path.join(self.root_folder, 'CelebA-HQ-img')

        image = Image.open(os.path.join(image_folder, self.image_list[index]))

        if self.transform:
            image = self.transform(image)

        label = self.label_list[index]

        return (image, torch.tensor(label, dtype=torch.float))
