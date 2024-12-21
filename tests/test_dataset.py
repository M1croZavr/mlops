import os
import unittest
from unittest.mock import Mock, patch

import torch

from models.dataset import Dataset


class TestDataset(unittest.TestCase):
    @patch("models.dataset.client")
    def test_dataset(self, client_mock):
        self.opened_fixtures = []
        client_mock.list_objects = self._list_objects_mock
        client_mock.get_object = self._get_object_mock

        self.split_dir = "train"
        dataset = Dataset(self.split_dir)
        example_image_tensor, example_target = dataset[0]

        self.assertIsInstance(example_image_tensor, torch.FloatTensor)
        self.assertEqual(example_image_tensor.shape, torch.Size((1, 28, 28)))
        self.assertEqual(len(dataset), 8)
        self.assertLessEqual(example_target, 9)
        self.assertGreaterEqual(example_target, 0)

    @staticmethod
    def _list_objects_mock(bucket_name, prefix):
        for file_path in os.listdir("tests/fixtures"):
            if file_path.startswith("mnist_example"):
                mock_object = Mock()
                mock_object.object_name = file_path
                yield mock_object

    def _get_object_mock(self, bucket_name, object_name):
        if object_name == f"{self.split_dir}/targets.pkl":
            response_object = open("tests/fixtures/targets.pkl", "rb")
        else:
            response_object = open(f"tests/fixtures/{object_name}", "rb")
        self.opened_fixtures.append(response_object)
        return response_object

    def tearDown(self):
        for response_object in self.opened_fixtures:
            response_object.close()


if __name__ == "__main__":
    unittest.main()
