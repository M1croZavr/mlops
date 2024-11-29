import io
import os
import pathlib
import tarfile
from typing import Optional, Annotated

import pytorch_lightning as pl
import torch
import uvicorn
from fastapi import FastAPI, Query, Path, Body, File, UploadFile, status, HTTPException
from pydantic import BaseModel, Field

from models.data_preprocessing import read_jpg_as_tensor
from models.models import LightningPerceptronClassifier, LightningCNNClassifier

ROOT = pathlib.Path(__file__).parent.parent.parent

app = FastAPI(title="F-app", version="0.1.0")


class BaseHyperparameters(BaseModel):
    epochs: int = Field(examples=[2])
    learning_rate: Optional[float] = Field(default=0.001, examples=[0.001])
    batch_size: Optional[int] = Field(default=32, examples=[32])


class PerceptronClassifierHyperparameters(BaseHyperparameters):
    hidden_dim: int = Field(examples=[64])


class CNNClassifierHyperparameters(BaseHyperparameters):
    hidden_channels: int = Field(examples=[16])


class AvailableModelDescription(BaseModel):
    class_name: str
    endpoint: str
    description: str


class ModelInferenceResult(BaseModel):
    label: int
    probability: float


class HealthCheck(BaseModel):
    status: str = Field(default="OK")


@app.post("/mnist/load_data", response_model=None, status_code=status.HTTP_201_CREATED, tags=["tar loader"])
async def load_mnist_dataset(
    train_dataset_file: Annotated[bytes, File(description="Training MNIST dataset tar.gz archive")]
):
    data_root = ROOT / "data"
    data_root.mkdir(exist_ok=True)

    io_train_dataset_bytes = io.BytesIO(train_dataset_file)
    tar = tarfile.open(mode="r:gz", fileobj=io_train_dataset_bytes)
    tar.extractall(path=data_root)
    return


@app.post("/mnist/fit_perceptron", response_model=None, status_code=status.HTTP_201_CREATED, tags=["fit models"])
async def fit_mnist_perceptron(
    hyperparameters: Annotated[
        PerceptronClassifierHyperparameters,
        Body(description="Perceptron model hyperparameters")
    ],
    model_filename: Annotated[str, Query(description="Filename for a model checkpoint *.ckpt")] = "perceptron"
):
    artifacts_dir = ROOT / "artifacts" / "mnist_perceptron_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if (artifacts_dir / f"{model_filename}.ckpt").exists():
        os.remove(artifacts_dir / f"{model_filename}.ckpt")

    data_dir = ROOT / "data" / "MNIST_DATA"
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data was not loaded, load data before fitting the model")

    model = LightningPerceptronClassifier(
        data_root=data_dir,
        input_dim=28 * 28,
        hidden_dim=hyperparameters.hidden_dim,
        output_dim=10,
        learning_rate=hyperparameters.learning_rate,
        batch_size=hyperparameters.batch_size
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=artifacts_dir,
        filename=model_filename,
        save_top_k=1,
        monitor="Validation loss"
    )
    trainer = pl.Trainer(
        max_epochs=hyperparameters.epochs,
        default_root_dir=artifacts_dir,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model)
    return


@app.post("/mnist/fit_cnn", response_model=None, status_code=status.HTTP_201_CREATED, tags=["fit models"])
async def fit_mnist_cnn(
    hyperparameters: Annotated[
        CNNClassifierHyperparameters,
        Body(description="Convolutional model hyperparameters")
    ],
    model_filename: Annotated[str, Query(description="Filename for a model checkpoint *.ckpt")] = "cnn"
):
    artifacts_dir = ROOT / "artifacts" / "mnist_cnn_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if (artifacts_dir / f"{model_filename}.ckpt").exists():
        os.remove(artifacts_dir / f"{model_filename}.ckpt")

    data_dir = ROOT / "data" / "MNIST_DATA"
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data was not loaded, load data before fitting the model")

    model = LightningCNNClassifier(
        data_root=data_dir,
        in_channels=1,
        hidden_channels=hyperparameters.hidden_channels,
        input_dim=28,
        output_dim=10,
        learning_rate=hyperparameters.learning_rate,
        batch_size=hyperparameters.batch_size
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=artifacts_dir,
        filename=model_filename,
        save_top_k=1,
        monitor="Validation loss"
    )
    trainer = pl.Trainer(
        max_epochs=hyperparameters.epochs,
        default_root_dir=artifacts_dir,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model)
    return


@app.get(
    "/available_models",
    response_model=list[AvailableModelDescription],
    status_code=status.HTTP_200_OK,
    tags=["available models"]
)
async def list_available_models() -> list[AvailableModelDescription]:
    available_models = [
        AvailableModelDescription(
            class_name="LightningPerceptronClassifier",
            endpoint="/mnist/fit_perceptron",
            description="Perceptron for classifying MNIST dataset samples"
        ),
        AvailableModelDescription(
            class_name="LightningCNNClassifier",
            endpoint="/mnist/fit_cnn",
            description="Convolutional neural network for classifying MNIST dataset samples"
        )
    ]
    return available_models


@app.post(
    "/mnist/predict_perceptron/{model_filename}",
    response_model=ModelInferenceResult,
    status_code=status.HTTP_200_OK,
    tags=["predict using models"]
)
async def predict_mnist_perceptron(
    model_filename: Annotated[str, Path(example="perceptron", title="Previously fitted model's name")],
    file: UploadFile
) -> ModelInferenceResult:
    artifacts_dir = ROOT / "artifacts" / "mnist_perceptron_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / f"{model_filename}.ckpt"

    if checkpoint_path.exists():
        model = LightningPerceptronClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
    else:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint of LightningPerceptronClassifier for provided model_filename not found"
        )

    x = read_jpg_as_tensor(file.file).to(model.device)
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    value, index = torch.max(probabilities, 1)
    model_inference_result = ModelInferenceResult(label=index.item(), probability=value.item())
    return model_inference_result


@app.post(
    "/mnist/predict_cnn/{model_filename}",
    response_model=ModelInferenceResult,
    status_code=status.HTTP_200_OK,
    tags=["predict using models"]
)
async def predict_mnist_cnn(
    model_filename: Annotated[str, Path(example="cnn", title="Previously fitted model's name")],
    file: UploadFile
) -> ModelInferenceResult:
    artifacts_dir = ROOT / "artifacts" / "mnist_cnn_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / f"{model_filename}.ckpt"

    if checkpoint_path.exists():
        model = LightningCNNClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
    else:
        raise HTTPException(
            status_code=404,
            detail="Checkpoint of LightningCNNClassifier for provided model_filename not found"
        )

    x = read_jpg_as_tensor(file.file).to(model.device)
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    value, index = torch.max(probabilities, 1)
    model_inference_result = ModelInferenceResult(label=index.item(), probability=value.item())
    return model_inference_result


@app.delete("/mnist/{model_filename}", response_model=None, status_code=status.HTTP_200_OK, tags=["delete models"])
def delete_checkpoint(
    model_filename: Annotated[str, Path(examples=["perceptron", "cnn"], title="Previously fitted model's name")]
):
    artifacts_dir = ROOT / "artifacts"
    model_filenames_paths = artifacts_dir.glob(f"*/{model_filename}.ckpt")
    if not model_filenames_paths:
        raise HTTPException(status_code=404, detail="Checkpoint for provided model_filename not found")
    else:
        for model_filename_path in model_filenames_paths:
            os.remove(model_filename_path)
    return


@app.get(
    "/health",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    tags=["healthcheck"],
    summary="Perform Health Check",
    response_description="Return HTTP Status Code 200 if the service is OK"
)
def perform_healthcheck() -> HealthCheck:
    try:
        ...
    except Exception:
        raise HTTPException(status_code=503)
    return HealthCheck(status="OK")


def main():
    """Launch uvicorn server"""
    uvicorn.run("src.fastapi_service.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
