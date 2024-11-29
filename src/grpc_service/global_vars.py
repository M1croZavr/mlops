import pathlib

MAX_MESSAGE_LENGTH = 20 * 1024 * 1024
DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "data"
MNIST_PATH = DATA_PATH / "MNIST_DATA.tar.gz"
