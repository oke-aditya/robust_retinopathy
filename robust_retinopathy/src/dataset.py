import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import config


__all__ = ["RetinopathyDataset"]


class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_name = os.path.join('../input/aptos2019-blindness-detection/train_images',
        #                         self.data.loc[idx, 'id_code'] + '.png')
        img_name = os.path.join(config.DATA_DIR, self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)
        # image = image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'], dtype=torch.long)

        if self.transforms is not None:
            img = self.transforms()(image)

        return {
            'image': img,
            'labels': label
        }
