from typing import BinaryIO

import torch
import torchvision
from PIL import Image


def read_jpg_as_tensor(file: BinaryIO) -> torch.Tensor:
    """Creates PIL image from jpg file and converts into tensor"""
    image = Image.open(file)
    tensor = torchvision.transforms.ToTensor()(image)
    return tensor
