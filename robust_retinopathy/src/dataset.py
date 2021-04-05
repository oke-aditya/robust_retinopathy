import torch
from torch.utils.data import Dataset
from PIL import Image
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

        image = Image.open(img_name)
        image = image.resize((224, 224), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'], dtype=torch.long)

        if self.transform is not None:
            img = self.transform(image)

        return (img, label)
