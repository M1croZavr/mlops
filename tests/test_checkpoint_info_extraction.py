import unittest
from unittest.mock import Mock, patch

from fastapi_service.s3.utils import get_all_checkpoints_info


class TestCheckpointsExtraction(unittest.TestCase):
    @patch("fastapi_service.s3.utils.client")
    def test_get_all_checkpoints_info(self, client_mock):
        client_mock.list_objects = self._list_objects_mock
        extracted_checkpoint_info = get_all_checkpoints_info()
        valid_result = [
            {
                "dataset_folder_name": "MNIST",
                "model_filename": "model",
            },
            {
                "dataset_folder_name": "CIFAR10",
                "model_filename": "model1",
            },
            {
                "dataset_folder_name": "CIFAR10",
                "model_filename": "model2",
            },
            {
                "dataset_folder_name": "IMAGENET",
                "model_filename": "checkpoint",
            },
        ]
        self.assertEqual(extracted_checkpoint_info, valid_result)

    @staticmethod
    def _list_objects_mock(bucket_name, prefix=None, recursive=False):
        directories = ["/MNIST", "/CIFAR10", "/IMAGENET"]
        checkpoints = [
            "/MNIST/model.ckpt",
            "/CIFAR10/model1.ckpt",
            "/CIFAR10/model2.ckpt",
            "/IMAGENET/checkpoint.ckpt",
        ]

        mock_objects = []
        if not recursive:
            for directory_path in directories:
                mock_object = Mock()
                mock_object.is_dir = True
                mock_object.object_name = directory_path
                mock_objects.append(mock_object)
        else:
            for checkpoint_path in checkpoints:
                if prefix and checkpoint_path.startswith(prefix):
                    mock_object = Mock()
                    mock_object.object_name = checkpoint_path
                    mock_objects.append(mock_object)

        return mock_objects


if __name__ == "__main__":
    unittest.main()
