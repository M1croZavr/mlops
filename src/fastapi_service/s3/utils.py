import random
from pathlib import Path
from tarfile import TarFile

from fastapi import HTTPException
from minio.error import S3Error
from PIL import Image
from torchvision.transforms.functional import get_image_num_channels
from tqdm import tqdm

from fastapi_service.config import ARTIFACTS_ROOT
from fastapi_service.s3 import (
    ARTIFACTS_BUCKET_NAME,
    DATASETS_BUCKET_NAME,
    SOURCES_BUCKET_NAME,
    client,
)


def extract_tar_into_s3(tar: TarFile):
    for tar_member in tqdm(tar.getmembers()):
        if tar_member.isfile():
            client.put_object(
                DATASETS_BUCKET_NAME,
                tar_member.name,
                tar.extractfile(tar_member),
                length=-1,
                part_size=5 * 1024 * 1024,
            )
    return


def get_dataset_dir(dataset_folder_name: str):
    if not any(client.list_objects(DATASETS_BUCKET_NAME, dataset_folder_name)):
        raise HTTPException(
            status_code=404,
            detail="Data was not found in s3 storage, load data before fitting the model",
        )
    else:
        return dataset_folder_name


def get_width_height_channels(dataset_folder_name: str):
    images_dir = f"{dataset_folder_name}/validation/images/"
    random_image_object = client.get_object(
        DATASETS_BUCKET_NAME,
        random.choice(
            list(client.list_objects(DATASETS_BUCKET_NAME, images_dir))
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
    model_filename = model_filename.removesuffix(".ckpt")
    client.fput_object(
        ARTIFACTS_BUCKET_NAME,
        f"{dataset_folder_name}/{model_filename}.ckpt",
        f"{local_checkpoint_path.resolve()}.ckpt",
    )
    return


def get_all_checkpoints_info():
    checkpoints_list = []
    for dataset_folder_name in client.list_objects(ARTIFACTS_BUCKET_NAME):
        if dataset_folder_name.is_dir:
            for model_filename in client.list_objects(
                ARTIFACTS_BUCKET_NAME,
                prefix=dataset_folder_name.object_name,
                recursive=True,
            ):
                if model_filename.object_name.endswith(".ckpt"):
                    checkpoints_list.append(
                        {
                            "dataset_folder_name": dataset_folder_name.object_name.strip(
                                "/"
                            ),
                            "model_filename": model_filename.object_name.removesuffix(
                                ".ckpt"
                            ).split("/")[-1],
                        }
                    )
    return checkpoints_list


def get_checkpoint_path(dataset_folder_name: str, model_filename: str):
    model_filename = model_filename.removesuffix(".ckpt")
    s3_checkpoint_path = f"{dataset_folder_name}/{model_filename}.ckpt"
    local_checkpoint_path = ARTIFACTS_ROOT / s3_checkpoint_path

    try:
        client.fget_object(
            ARTIFACTS_BUCKET_NAME, s3_checkpoint_path, local_checkpoint_path
        )
    except S3Error:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint of Lightning Model for provided model_filename not found in S3 storage",
        )

    return local_checkpoint_path


def delete_checkpoint_from_storage(model_filename: str):
    model_filename = model_filename.removesuffix(".ckpt")
    model_filename = f"{model_filename}.ckpt"

    removed = False
    for object_info in client.list_objects(ARTIFACTS_BUCKET_NAME, recursive=True):
        if object_info.object_name.endswith(model_filename):
            client.remove_object(ARTIFACTS_BUCKET_NAME, object_info.object_name)
            removed = True

    if not removed:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint for provided model_filename not found in s3 storage",
        )

    return {}


def create_buckets():
    # Make sources bucket
    if not client.bucket_exists(SOURCES_BUCKET_NAME):
        client.make_bucket(SOURCES_BUCKET_NAME)
    # Make datasets bucket
    if not client.bucket_exists(DATASETS_BUCKET_NAME):
        client.make_bucket(DATASETS_BUCKET_NAME)
    # Make artifacts bucket
    if not client.bucket_exists(ARTIFACTS_BUCKET_NAME):
        client.make_bucket(ARTIFACTS_BUCKET_NAME)
