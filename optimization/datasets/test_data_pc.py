import os
import torch
from torch.utils.data import Dataset
import numpy as np

class PointCloud(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)

        points = torch.from_numpy(np.loadtxt(file_path)).float()
        points = points.view(-1, 3)  # Reshape points to (X, 3)

        return {'points': points, 'file_name': str(file_name[:-4])}