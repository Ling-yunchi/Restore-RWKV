import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HYPSO1_Dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None):
        self.root_path = root_path
        self.train = train
        self.transform = transform
        self.train_ratio = 0.8

        self.data_paths = []
        self.label_paths = []

        for subdir in os.listdir(root_path):
            subdir_path = os.path.join(root_path, subdir)
            if os.path.isdir(subdir_path):
                data_dir = os.path.join(subdir_path, 'DATA')
                label_dir = os.path.join(subdir_path, 'GROUND-TRUTH LABELS')

                if not os.path.exists(data_dir) or not os.path.exists(label_dir):
                    continue

                # 查找 DATA 文件夹中的 .npy 文件
                data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
                # 查找 GROUND-TRUTH LABELS 文件夹中的 .npy 文件
                label_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]

                # 确保每个子文件夹中都有一个 .npy 文件
                if data_files and len(data_files) == 1 and label_files and len(label_files) == 1:
                    data_file_path = os.path.join(data_dir, data_files[0])
                    label_file_path = os.path.join(label_dir, label_files[0])
                    self.data_paths.append(data_file_path)
                    self.label_paths.append(label_file_path)

        if self.train:
            self.data_paths = self.data_paths[:int(len(self.data_paths) * self.train_ratio)]
            self.label_paths = self.label_paths[:int(len(self.label_paths) * self.train_ratio)]
        else:
            self.data_paths = self.data_paths[int(len(self.data_paths) * self.train_ratio):]
            self.label_paths = self.label_paths[int(len(self.label_paths) * self.train_ratio):]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path, label_path = self.data_paths[idx], self.label_paths[idx]

        # Load data and label
        np_data = np.load(data_path)
        data = torch.tensor(np_data, dtype=torch.float64)
        data = data.permute(2, 0, 1)

        np_label = np.load(label_path)
        label = torch.tensor(np_label, dtype=torch.long)
        label = label.unsqueeze(0)
        label = torch.nn.functional.one_hot(label, num_classes=3)
        label = label.squeeze(0).permute(2, 0, 1)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        return data, label


if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
    ])
    dataset = HYPSO1_Dataset(root_path='../dataset/1-DATA WITH GROUND-TRUTH LABELS')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Example of iterating through the dataloader
    for data, label in dataloader:
        # size batch_size x channels x height x width
        # print label channel all values
        print(data.size(), label.size())
