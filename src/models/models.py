from pathlib import Path

import idx2numpy
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class LightningPerceptronClassifier(pl.LightningModule):
    def __init__(
        self,
        data_root: Path,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        super(LightningPerceptronClassifier, self).__init__()

        self.data_root = data_root

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
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flattener(x)

        output = self.relu(self.bn1(self.fc1(x)))
        output = self.relu(self.bn2(self.fc2(output)))
        output = self.fc3(output)

        return output

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True
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
            dataset, batch_size=self.batch_size, num_workers=2, persistent_workers=True
        )
        return loader


if __name__ == "__main__":
    # dataset = torchvision.datasets.MNIST(
    #     self.data_path,
    #     train=False,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=False
    # )
    model = LightningPerceptronClassifier(
        Path(__file__).parent.parent.parent / "data" / "MNIST_DATA",
        28 * 28,
        32,
        10
    )
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=None)
    trainer.fit(model)
