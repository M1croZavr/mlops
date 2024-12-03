import os
import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: Path):
        self.images_dir = split_dir / "images"
        self.images_files = os.listdir(self.images_dir)

        self.target_pkl_file = split_dir / "targets.pkl"
        with open(self.target_pkl_file, "rb") as file:
            self.targets = pickle.load(file)

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item: int):
        image = Image.open(self.images_dir / self.images_files[item])
        target = self.targets[self.images_files[item]]

        return self.transforms(image), target

    def __len__(self):
        return len(self.images_files)
