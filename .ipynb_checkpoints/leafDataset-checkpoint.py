import os
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

class LeafDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label