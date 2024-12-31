from typing import BinaryIO

import torch
import torchvision
from PIL import Image


def read_jpg_as_tensor(file: BinaryIO) -> torch.Tensor:
    """Converts file to a tensor.

    Creates PIL image from jpg file and converts into tensor.

    Parameters
    ----------
    file : BinaryIO
        Binary object of input .jpg image.

    Returns
    -------
    torch.Tensor
        Image's pytorch tensor.
    """
    image = Image.open(file)
    tensor = torchvision.transforms.ToTensor()(image)
    return tensor
