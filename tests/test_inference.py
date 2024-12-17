import unittest

import pytorch_lightning as pl
from torch import nn

from models.inference import predict


class MockModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28 * 1, 10), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class MockImage:
    def __init__(self):
        self.bytesio = open("tests/fixtures/mnist_example.jpg", "rb")

    @property
    def file(self):
        return self.bytesio


class TestInference(unittest.TestCase):
    def setUp(self):
        self.mock_model = MockModel()
        self.mock_image = MockImage()

    def test_type(self):
        index, value = predict(self.mock_model, self.mock_image)
        self.assertIsInstance(index, int)
        self.assertIsInstance(value, float)

    def test_value(self):
        _, value = predict(self.mock_model, self.mock_image)
        self.assertLessEqual(value, 1)
        self.assertGreaterEqual(value, 0)

    def test_index(self):
        index, _ = predict(self.mock_model, self.mock_image)
        self.assertLessEqual(index, 9)
        self.assertGreaterEqual(index, 0)

    def tearDown(self):
        self.mock_image.bytesio.close()


if __name__ == "__main__":
    unittest.main()
