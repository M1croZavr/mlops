import io
import os
import tarfile
from typing import Annotated

import psutil
import uvicorn
from fastapi import Body, FastAPI, File, Path, Query, UploadFile, status
from tqdm import tqdm

from fastapi_service.config import APP_PORT, APP_VERSION, ARTIFACTS_ROOT, CLASS_LABELS
from fastapi_service.models import (
    AvailableCheckpointDescription,
    AvailableModelDescription,
    CNNClassifierHyperparameters,
    HealthCheck,
    ModelInferenceResult,
    PerceptronClassifierHyperparameters,
)
from fastapi_service.s3 import client, datasets_bucket_name
from fastapi_service.s3.utils import (
    delete_checkpoint_from_storage,
    get_all_checkpoints_info,
    get_checkpoint_path,
)
from models import inference, train
from models.modules import LightningCNNClassifier, LightningPerceptronClassifier

app = FastAPI(title="F-app", version=APP_VERSION)


@app.post("/load_data", status_code=status.HTTP_201_CREATED, tags=["load dataset"])
async def load_dataset(
    dataset_file: Annotated[
        bytes,
        File(
            description="Dataset of tar.gz archive format which consists of `train` and `validation` folders"
        ),
    ],
):
    """Loads tar.gz archive from request and extract into s3 object storage.

    Note that dataset is identified by archive name (root directory name e.g. MNIST, CIFAR10)
    and should have `train` and `validation` folders inside
    which consist of pictures PIL.Image readable format.

    Parameters
    ----------
    dataset_file : bytes
        Archive file bytes.

    Returns
    -------
    dict
    """
    io_dataset_bytes = io.BytesIO(dataset_file)
    with tarfile.open(mode="r:gz", fileobj=io_dataset_bytes) as tar:
        for tar_member in tqdm(tar.getmembers()):
            if tar_member.isfile():
                client.put_object(
                    datasets_bucket_name,
                    tar_member.name,
                    tar.extractfile(tar_member),
                    length=-1,
                    part_size=5 * 1024 * 1024,
                )

    return {}


@app.post("/fit/perceptron", status_code=status.HTTP_201_CREATED, tags=["fit model"])
async def fit_perceptron(
    hyperparameters: Annotated[
        PerceptronClassifierHyperparameters,
        Body(description="Perceptron model hyperparameters"),
    ],
    dataset_folder_name: Annotated[
        str,
        Query(description="Folder name for a dataset", examples=["MNIST", "CIFAR10"]),
    ] = "MNIST",
    model_filename: Annotated[
        str, Query(description="Filename for a model checkpoint {model_filename}.ckpt")
    ] = "perceptron",
):
    model = LightningPerceptronClassifier(
        dataset_folder_name=dataset_folder_name,
        hidden_dim=hyperparameters.hidden_dim,
        output_dim=hyperparameters.n_classes,
        learning_rate=hyperparameters.learning_rate,
        batch_size=hyperparameters.batch_size,
    )
    train.train(model, hyperparameters.epochs, dataset_folder_name, model_filename)

    return {}


@app.post("/fit/cnn", status_code=status.HTTP_201_CREATED, tags=["fit model"])
async def fit_cnn(
    hyperparameters: Annotated[
        CNNClassifierHyperparameters,
        Body(description="Convolutional model hyperparameters"),
    ],
    dataset_folder_name: Annotated[
        str,
        Query(description="Folder name for a dataset", examples=["MNIST", "CIFAR10"]),
    ] = "MNIST",
    model_filename: Annotated[
        str, Query(description="Filename for a model checkpoint {model_filename}.ckpt")
    ] = "cnn",
):
    model = LightningCNNClassifier(
        dataset_folder_name=dataset_folder_name,
        hidden_channels=hyperparameters.hidden_channels,
        output_dim=hyperparameters.n_classes,
        learning_rate=hyperparameters.learning_rate,
        batch_size=hyperparameters.batch_size,
    )
    train.train(model, hyperparameters.epochs, dataset_folder_name, model_filename)

    return {}


@app.get(
    "/list/available_models",
    response_model=list[AvailableModelDescription],
    status_code=status.HTTP_200_OK,
    tags=["available models"],
)
async def list_available_models() -> list[AvailableModelDescription]:
    available_models = [
        AvailableModelDescription(
            class_name="LightningPerceptronClassifier",
            endpoint="/fit/perceptron",
            description="Perceptron for classifying",
        ),
        AvailableModelDescription(
            class_name="LightningCNNClassifier",
            endpoint="/fit/cnn",
            description="Convolutional neural network for classifying",
        ),
    ]
    return available_models


@app.get(
    "/list/available_checkpoints",
    response_model=list[AvailableCheckpointDescription],
    status_code=status.HTTP_200_OK,
    tags=["available models checkpoints"],
)
async def list_available_checkpoints() -> list[AvailableCheckpointDescription]:
    checkpoints_list = get_all_checkpoints_info()
    available_checkpoint = [
        AvailableCheckpointDescription(**checkpoint_info)
        for checkpoint_info in checkpoints_list
    ]
    return available_checkpoint


@app.post(
    "/predict/{model_filename}",
    response_model=ModelInferenceResult,
    status_code=status.HTTP_200_OK,
    tags=["predict"],
)
async def predict(
    model_filename: Annotated[
        str, Path(example="perceptron", title="Previously fitted model's name")
    ],
    image_file: UploadFile,
    dataset_folder_name: Annotated[
        str,
        Query(description="Folder name for a dataset", examples=["MNIST", "CIFAR10"]),
    ] = "MNIST",
) -> ModelInferenceResult:
    model_checkpoint_path = get_checkpoint_path(dataset_folder_name, model_filename)
    try:
        model = LightningCNNClassifier.load_from_checkpoint(model_checkpoint_path)
    except TypeError:
        model = LightningPerceptronClassifier.load_from_checkpoint(
            model_checkpoint_path
        )

    index, value = inference.predict(model, image_file)
    model_inference_result = ModelInferenceResult(
        label=CLASS_LABELS.get(dataset_folder_name)[index], probability=value
    )
    return model_inference_result


@app.delete(
    "/{model_filename}",
    status_code=status.HTTP_200_OK,
    tags=["delete model checkpoint"],
)
async def delete_checkpoint(
    model_filename: Annotated[
        str,
        Path(examples=["perceptron", "cnn"], title="Previously fitted model's name"),
    ],
):
    # Remove locally if exists
    model_filenames_paths = ARTIFACTS_ROOT.glob(f"*/{model_filename.rstrip('.ckpt')}.ckpt")
    if model_filenames_paths:
        for model_filename_path in model_filenames_paths:
            os.remove(model_filename_path)

    # Remove from object storage
    delete_checkpoint_from_storage(model_filename)

    return {}


@app.get(
    "/health",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    tags=["healthcheck"],
    summary="Perform Health Check",
    response_description="Return HTTP Status Code 200 if the service is OK",
)
async def perform_healthcheck() -> HealthCheck:
    cpu_usage_percent = psutil.cpu_percent()
    ram_usage_percent = psutil.virtual_memory().percent
    return HealthCheck(
        status="OK",
        cpu_usage_percent=cpu_usage_percent,
        ram_usage_percent=ram_usage_percent,
    )


def main():
    """Launch uvicorn server on specified host and port"""
    uvicorn.run(
        "src.fastapi_service.main:app", host="127.0.0.1", port=APP_PORT, reload=True
    )


if __name__ == "__main__":
    main()
