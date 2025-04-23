# utils/dataset_loader.py

import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dirs, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dirs = image_dirs
        self.transform = transform

        # Create a mapping from label name to numeric class
        self.label_map = {label: idx for idx, label in enumerate(self.data['dx'].unique())}
        self.data['label'] = self.data['dx'].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id'] + '.jpg'
        img_path = self.find_image(img_name)
        image = Image.open(img_path).convert("RGB")

        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

    def find_image(self, image_name):
        for directory in self.image_dirs:
            full_path = os.path.join(directory, image_name)
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError(f"{image_name} not found in {self.image_dirs}")
