import os
import random

from fastapi import HTTPException
from PIL import Image
from torchvision.transforms.functional import get_image_num_channels

from fastapi_service.config import ARTIFACTS_ROOT, DATA_ROOT


def get_checkpoint_dir_and_name(dataset_folder_name: str, model_filename: str):
    checkpoint_dir = ARTIFACTS_ROOT / dataset_folder_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = model_filename

    if (checkpoint_dir / checkpoint_name).exists():
        os.remove(checkpoint_dir / checkpoint_name)

    return checkpoint_dir, checkpoint_name


def get_dataset_dir(dataset_folder_name: str):
    dataset_dir = DATA_ROOT / dataset_folder_name
    if not dataset_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Data was not loaded, load data before fitting the model",
        )
    else:
        return dataset_dir


def get_all_checkpoints_info():
    checkpoints_list = []
    for dataset_folder_name in os.listdir(ARTIFACTS_ROOT):
        for model_filename in (ARTIFACTS_ROOT / dataset_folder_name).glob("*.ckpt"):
            checkpoints_list.append(
                {
                    "dataset_folder_name": dataset_folder_name,
                    "model_filename": model_filename.name.rstrip(".ckpt"),
                }
            )
    return checkpoints_list


def get_checkpoint_path(dataset_folder_name: str, model_filename: str):
    artifacts_dir = ARTIFACTS_ROOT / dataset_folder_name
    model_filename = model_filename.rstrip(".ckpt")
    checkpoint_path = artifacts_dir / f"{model_filename}.ckpt"

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Checkpoint of Lightning Model for provided model_filename not found",
        )

    return checkpoint_path


def get_width_height_channels(dataset_folder_name: str):
    images_dir = DATA_ROOT / dataset_folder_name / "train" / "images"
    random_train_image = Image.open(images_dir / random.choice(os.listdir(images_dir)))
    return (
        random_train_image.width,
        random_train_image.height,
        get_image_num_channels(random_train_image),
    )
