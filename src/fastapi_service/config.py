import pathlib

APP_VERSION = "0.1.0"
APP_PORT = 8000

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
