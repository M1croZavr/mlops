import io
import pathlib
import shutil
import tarfile
from enum import Enum
from typing import Union, Optional, Annotated

import pytorch_lightning as pl
import uvicorn
from fastapi import FastAPI, Query, Path, Body, File, UploadFile, status
from pydantic import BaseModel, Field

from models.models import LightningPerceptronClassifier

app = FastAPI()


class PerceptronClassifierHyperparameters(BaseModel):
    epochs: int = Field(examples=[2])
    hidden_dim: int = Field(examples=[64])
    learning_rate: Optional[float] = Field(default=0.001, examples=[0.001])
    batch_size: Optional[int] = Field(default=32, examples=[32])


@app.post("/mnist/load_data", response_model=None, status_code=status.HTTP_201_CREATED)
async def load_mnist_dataset(
    train_dataset_file: Annotated[bytes, File(description="Training MNIST dataset tar.gz archive")]
):
    data_root = pathlib.Path(__file__).parent.parent.parent / "data"
    data_root.mkdir(exist_ok=True)

    io_train_dataset_bytes = io.BytesIO(train_dataset_file)
    tar = tarfile.open(mode="r:gz", fileobj=io_train_dataset_bytes)
    tar.extractall(path=data_root)
    return


@app.post("/mnist/fit_model", response_model=None, status_code=status.HTTP_201_CREATED)
async def fit_mnist_model(
    hyperparameters: Annotated[PerceptronClassifierHyperparameters, Body(description="Classifier hyperparameters")],
    model_filename: Annotated[str, Query(description="Filename for a model checkpoint *.ckpt")] = "mnist_classifier"
):
    root = pathlib.Path(__file__).parent.parent.parent
    artifacts_dir = root / "artifacts" / "mnist_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(artifacts_dir)

    model = LightningPerceptronClassifier(
        data_root=root / "data" / "MNIST_DATA",
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
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model)
    return


def main():
    """Launched uvicorn server"""
    uvicorn.run("src.backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
