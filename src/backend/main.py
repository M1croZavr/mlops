from enum import Enum
from typing import Union, Optional, Annotated

import uvicorn
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel

app = FastAPI()


class Hyperparameters(BaseModel):
    epochs: Optional[int] = 25
    hidden_dim: Optional[int] = 32
    learning_rate: Optional[float] = 0.001


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/")
def index():
    return "Hello, world!"


@app.post("/fit/{model_name}")
def fit_model(
        model_name: Annotated[ModelName, Path(title="Model to be fitted")],
        hyperparameters: Hyperparameters,
        q: Annotated[Union[str, None], Query(min_length=5, max_length=10)] = None
):
    if model_name is ModelName.resnet:
        return [model_name.name, model_name.value, q, hyperparameters]


def main():
    """Launched uvicorn server"""
    uvicorn.run("src.backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
