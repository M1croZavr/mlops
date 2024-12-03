import torch
from fastapi import UploadFile
from torch import nn

from models.data_preprocessing import read_jpg_as_tensor


def predict(model: nn.Module, image_file: UploadFile):
    model.eval()
    x = read_jpg_as_tensor(image_file.file).to(model.device).unsqueeze(dim=0)
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        value, index = torch.max(probabilities, 1)
    return index.item(), value.item()
