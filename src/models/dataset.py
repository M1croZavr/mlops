import pickle

import torch
from PIL import Image
from torchvision import transforms

from fastapi_service.s3 import client, datasets_bucket_name


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str):
        self.images_dir = f"{split_dir}/images/"
        self.images_files = list(
            client.list_objects(datasets_bucket_name, self.images_dir)
        )

        self.target_pkl_file = f"{split_dir}/targets.pkl"
        self.targets = pickle.load(
            client.get_object(datasets_bucket_name, self.target_pkl_file)
        )

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item: int):
        image_filename = self.images_files[item].object_name
        image = Image.open(client.get_object(datasets_bucket_name, image_filename))
        target = self.targets[image_filename.split("/")[-1]]

        return self.transforms(image), target

    def __len__(self):
        return len(self.images_files)
