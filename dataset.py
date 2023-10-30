import torch
import numpy
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io

class DebrisDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len_(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[item, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[item, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)