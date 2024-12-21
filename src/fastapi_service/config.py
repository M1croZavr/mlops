import os
import pathlib

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if exists

APP_VERSION = "0.1.0"
# Specify this host inside container to reach uvicorn on localhost
APP_HOST = "0.0.0.0"
APP_PORT = os.getenv("APP_PORT", 8000)

MLFLOW_PORT = os.getenv("MLFLOW_PORT", 9000)

ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_ROOT = ROOT / "data"
ARTIFACTS_ROOT = ROOT / "artifacts"

CLASS_LABELS = {
    "CIFAR10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "MNIST": list(range(10)),
}
