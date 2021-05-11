import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os


__all__ = ["RetinopathyDataset"]


class RetinopathyDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_name = os.path.join('../input/aptos2019-blindness-detection/train_images',
        #                         self.data.loc[idx, 'id_code'] + '.png')

        img_name = os.path.join(self.image_dir, self.data.loc[idx, 'id_code'] + '.png')

        tensor_image = read_image(img_name)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'], dtype=torch.long)

        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()

        if self.transforms is not None:
            tensor_image = self.transforms(tensor_image)

        return (tensor_image, label)
