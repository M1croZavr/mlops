from typing import Optional, Union

from pydantic import BaseModel, Field


class BaseHyperparameters(BaseModel):
    epochs: int = Field(examples=[2])
    n_classes: int = Field(examples=[10])
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


class AvailableCheckpointDescription(BaseModel):
    dataset_name: str
    model_filename: str


class ModelInferenceResult(BaseModel):
    label: Union[int, str]
    probability: float


class HealthCheck(BaseModel):
    status: str = Field(default="OK")
    cpu_usage_percent: float
    ram_usage_percent: float
