import unittest

import torch

from models.data_preprocessing import read_jpg_as_tensor


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.example1 = open("tests/fixtures/mnist_example1.jpg", "rb")
        self.example2 = open("tests/fixtures/cifar10_example.jpg", "rb")

    def test_type(self):
        self.assertIsInstance(read_jpg_as_tensor(self.example1), torch.FloatTensor)
        self.assertIsInstance(read_jpg_as_tensor(self.example2), torch.FloatTensor)

    def test_size(self):
        self.assertEqual(
            read_jpg_as_tensor(self.example1).shape, torch.Size((1, 28, 28))
        )
        self.assertEqual(
            read_jpg_as_tensor(self.example2).shape, torch.Size((3, 32, 32))
        )

    def tearDown(self):
        self.example1.close()
        self.example2.close()


if __name__ == "__main__":
    unittest.main()
