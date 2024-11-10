from pathlib import Path

import idx2numpy
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LightningBaseMnistModule(pl.LightningModule):
    """Base MNIST LightningModule"""
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 32
    ):
        super(LightningBaseMnistModule, self).__init__()

        self.data_root = data_root
        self.batch_size = batch_size

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("Training loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        images_file = str(self.data_root / "train-images-idx3-ubyte")
        images_tensor = torch.from_numpy(idx2numpy.convert_from_file(images_file).copy()) / 255
        labels_file = str(self.data_root / "train-labels-idx1-ubyte")
        labels_tensor = torch.from_numpy(idx2numpy.convert_from_file(labels_file).copy()).to(torch.int64)

        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True
        )
        return loader

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("Validation loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        images_file = str(self.data_root / "t10k-images-idx3-ubyte")
        images_tensor = torch.from_numpy(idx2numpy.convert_from_file(images_file).copy()) / 255
        labels_file = str(self.data_root / "t10k-labels-idx1-ubyte")
        labels_tensor = torch.from_numpy(idx2numpy.convert_from_file(labels_file).copy()).to(torch.int64)

        dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True
        )
        return loader


class LightningPerceptronClassifier(LightningBaseMnistModule):
    """Perceptron classifier"""
    def __init__(
        self,
        data_root: Path,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        super(LightningPerceptronClassifier, self).__init__(data_root=data_root, batch_size=batch_size)

        self.fc1 = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim
        )
        self.fc2 = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim * 2
        )
        self.fc3 = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=output_dim
        )

        self.bn1 = nn.BatchNorm1d(
            num_features=hidden_dim
        )
        self.bn2 = nn.BatchNorm1d(
            num_features=hidden_dim * 2
        )

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


class LightningCNNClassifier(LightningBaseMnistModule):
    """Convolutional neural network classifier"""
    def __init__(
        self,
        data_root: Path,
        in_channels: int,
        hidden_channels: int,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        super(LightningCNNClassifier, self).__init__(data_root=data_root, batch_size=batch_size)

        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )
        self.c2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )
        self.c3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            padding=1
        )
        self.c4 = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2,
            kernel_size=3,
            padding=1
        )

        self.p1 = nn.MaxPool2d(2, 2)
        self.p2 = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(
            num_features=hidden_channels
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=hidden_channels
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=hidden_channels * 2
        )
        self.bn4 = nn.BatchNorm2d(
            num_features=hidden_channels * 2
        )

        self.fc = nn.Linear(
            in_features=int((input_dim / 4) ** 2 * hidden_channels * 2),
            out_features=output_dim
        )

        self.relu = nn.ReLU()

        self.flattener = nn.Flatten()

        self.lr = learning_rate

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.relu(self.bn1(self.c1(x.unsqueeze(dim=1))))
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
    model = LightningCNNClassifier(
        Path(__file__).parent.parent.parent / "data" / "MNIST_DATA",
        1,
        16,
        28,
        10
    )
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=None)
    trainer.fit(model)
