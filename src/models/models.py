import pytorch_lightning as pl
import torch
from torch import nn


class LightningPerceptronClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(LightningPerceptronClassifier, self).__init__()

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

    def forward(self, x: torch.Tensor):
        x = self.flattener(x)

        output = self.relu(self.bn1(self.fc1(x)))
        output = self.relu(self.bn2(self.fc2(output)))
        output = self.fc3(output)

        return output


if __name__ == '__main__':
    model = LightningPerceptronClassifier(16, 32, 3)
    t = torch.randn((32, 1, 4, 4))
    result = model(t)
    assert result.shape == torch.Size([32, 3]), 'Failed'
