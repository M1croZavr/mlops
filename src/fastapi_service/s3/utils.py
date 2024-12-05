import random
from pathlib import Path

from fastapi import HTTPException
from minio.error import S3Error
from PIL import Image
from torchvision.transforms.functional import get_image_num_channels

from fastapi_service.config import ARTIFACTS_ROOT
from fastapi_service.s3 import artifacts_bucket_name, client, datasets_bucket_name


def get_dataset_dir(dataset_folder_name: str):
    if not any(client.list_objects(datasets_bucket_name, dataset_folder_name)):
        raise HTTPException(
            status_code=404,
            detail="Data was not found in s3 storage, load data before fitting the model",
        )
    else:
        return dataset_folder_name


def get_width_height_channels(dataset_folder_name: str):
    images_dir = f"{dataset_folder_name}/validation/images/"
    random_image_object = client.get_object(
        datasets_bucket_name,
        random.choice(
            list(client.list_objects(datasets_bucket_name, images_dir))
        ).object_name,
    )
    random_train_image = Image.open(random_image_object)
    return (
        random_train_image.width,
        random_train_image.height,
        get_image_num_channels(random_train_image),
    )


def save_checkpoint(
    local_checkpoint_path: Path, dataset_folder_name: str, model_filename: str
):
    client.fput_object(
        artifacts_bucket_name,
        f"{dataset_folder_name}/{model_filename}.ckpt",
        f"{local_checkpoint_path.resolve()}.ckpt",
    )
    return


def get_all_checkpoints_info():
    checkpoints_list = []
    for dataset_folder_name in client.list_objects(artifacts_bucket_name):
        if dataset_folder_name.is_dir:
            for model_filename in client.list_objects(
                artifacts_bucket_name,
                prefix=dataset_folder_name.object_name,
                recursive=True,
            ):
                if model_filename.object_name.endswith(".ckpt"):
                    checkpoints_list.append(
                        {
                            "dataset_folder_name": dataset_folder_name.object_name.strip(
                                "/"
                            ),
                            "model_filename": model_filename.object_name.rstrip(
                                ".ckpt"
                            ).split("/")[-1],
                        }
                    )
    return checkpoints_list


def get_checkpoint_path(dataset_folder_name: str, model_filename: str):
    model_filename = model_filename.rstrip(".ckpt")
    s3_checkpoint_path = f"{dataset_folder_name}/{model_filename}.ckpt"
    local_checkpoint_path = ARTIFACTS_ROOT / s3_checkpoint_path

    try:
        client.fget_object(
            artifacts_bucket_name, s3_checkpoint_path, local_checkpoint_path
        )
    except S3Error:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint of Lightning Model for provided model_filename not found in S3 storage",
        )

    return local_checkpoint_path


def delete_checkpoint_from_storage(model_filename: str):
    model_filename = model_filename.rstrip(".ckpt")
    model_filename = f"{model_filename}.ckpt"

    removed = False
    for object_info in client.list_objects(artifacts_bucket_name, recursive=True):
        if object_info.object_name.endswith(model_filename):
            client.remove_object(artifacts_bucket_name, object_info.object_name)
            removed = True

    if not removed:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint for provided model_filename not found in s3 storage",
        )

    return {}
