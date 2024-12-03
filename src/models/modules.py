import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from fastapi_service.utils import get_dataset_dir, get_width_height_channels
from models.dataset import Dataset


class LightningBaseModule(pl.LightningModule):
    """Base LightningModule"""

    def __init__(self, dataset_folder_name: str, batch_size: int = 32):
        super(LightningBaseModule, self).__init__()

        self.dataset_dir = get_dataset_dir(dataset_folder_name)
        self.batch_size = batch_size

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("Training loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = Dataset(self.dataset_dir / "train")
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )
        return loader

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("Validation loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        validation_dataset = Dataset(self.dataset_dir / "validation")
        loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )
        return loader


class LightningPerceptronClassifier(LightningBaseModule):
    """Perceptron classifier"""

    def __init__(
        self,
        dataset_folder_name: str,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        super(LightningPerceptronClassifier, self).__init__(
            dataset_folder_name=dataset_folder_name, batch_size=batch_size
        )

        images_width, images_height, images_channels = get_width_height_channels(
            dataset_folder_name
        )

        self.fc1 = nn.Linear(
            in_features=images_width * images_height * images_channels,
            out_features=hidden_dim,
        )
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2)
        self.fc3 = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim * 2)

        self.relu = nn.ReLU()

        self.flattener = nn.Flatten()

        self.lr = learning_rate

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flattener(x)

        output = self.relu(self.bn1(self.fc1(x)))
        output = self.relu(self.bn2(self.fc2(output)))
        output = self.fc3(output)

        return output

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LightningCNNClassifier(LightningBaseModule):
    """Convolutional neural network classifier"""

    def __init__(
        self,
        dataset_folder_name: str,
        hidden_channels: int,
        output_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        super(LightningCNNClassifier, self).__init__(
            dataset_folder_name=dataset_folder_name, batch_size=batch_size
        )

        images_width, images_height, images_channels = get_width_height_channels(
            dataset_folder_name
        )

        self.c1 = nn.Conv2d(
            in_channels=images_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.c2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.c3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            padding=1,
        )
        self.c4 = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            padding=1,
        )

        self.p1 = nn.MaxPool2d(2, 2)
        self.p2 = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channels)
        self.bn3 = nn.BatchNorm2d(num_features=hidden_channels * 2)
        self.bn4 = nn.BatchNorm2d(num_features=hidden_channels * 2)

        self.fc = nn.Linear(
            in_features=int((images_width / 4) ** 2 * hidden_channels * 2),
            out_features=output_dim,
        )

        self.relu = nn.ReLU()

        self.flattener = nn.Flatten()

        self.lr = learning_rate

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.relu(self.bn1(self.c1(x)))
        output = self.relu(self.bn2(self.c2(output)))
        output = self.p1(output)

        output = self.relu(self.bn3(self.c3(output)))
        output = self.relu(self.bn4(self.c4(output)))
        output = self.p2(output)

        output = self.flattener(output)
        output = self.fc(output)

        return output

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    pass
    # dataset = torchvision.datasets.MNIST(
    #     self.data_path,
    #     train=False,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=False
    # )
    # model = LightningPerceptronClassifier(
    #     Path(__file__).parent.parent.parent / "data" / "MNIST_DATA",
    #     28 * 28,
    #     32,
    #     10
    # )
    # model = LightningCNNClassifier(
    #     Path(__file__).parent.parent.parent / "data" / "MNIST_DATA",
    #     1,
    #     16,
    #     28,
    #     10
    # )
    # trainer = pl.Trainer(fast_dev_run=True, default_root_dir=None)
    # trainer.fit(model)
