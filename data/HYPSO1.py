import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.MedicalDataUniform import DataSampler


class SameTransform:
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, img, mask):
        seed = torch.randint(0, 2 ** 32, (1,)).item()  # 随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transforms(img)
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transforms(mask)
        return img, mask


class HYPSO1_Dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None, _random=False):
        self.root_path = root_path
        self.train = train
        self.transform = SameTransform(transform) if transform else None
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

        if _random:
            random.shuffle(self.data_paths)
            random.shuffle(self.label_paths)

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
            data, label = self.transform(data, label)

        return data, label


class HYPSO1_PNG_Dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None):
        self.root_path = root_path
        self.train = train
        self.transform = SameTransform(transform) if transform else None
        self.train_ratio = 0.8

        self.data_paths = []
        self.label_paths = []

        for subdir in os.listdir(root_path):
            subdir_path = os.path.join(root_path, subdir)
            if os.path.isdir(subdir_path):
                label_dir = os.path.join(subdir_path, 'GROUND-TRUTH LABELS')

                if not os.path.exists(label_dir):
                    continue

                # 查找 GROUND-TRUTH LABELS 文件夹中的 -unbinned-converted-png.png 文件
                data_files = [f for f in os.listdir(label_dir) if f.endswith("-unbinned-converted-png.png")]
                # 查找 GROUND-TRUTH LABELS 文件夹中的 _labels.png 文件
                label_files = [f for f in os.listdir(label_dir) if f.endswith("_labels.png")]

                # 确保每个子文件夹中都有一个 .npy 文件
                if data_files and len(data_files) == 1 and label_files and len(label_files) == 1:
                    data_file_path = os.path.join(label_dir, data_files[0])
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
        data = Image.open(data_path)
        label = Image.open(label_path)

        if self.transform:
            data, label = self.transform(data, label)

        return data, label


def test_HYPSO1_Dataset():
    dataset = HYPSO1_Dataset(root_path='../dataset/1-DATA WITH GROUND-TRUTH LABELS')
    dataloader = DataLoader(dataset, batch_size=1)

    (data, label) = next(iter(dataloader))
    print(data.shape, label.shape)


def test_HYPSO1_PNG_Dataset():
    img_size = (256, 256)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(img_size),
        transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in [0, 90, 180, 270]]),
    ])
    dataset = HYPSO1_PNG_Dataset(root_path='../dataset/1-DATA WITH GROUND-TRUTH LABELS', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    datasampler = DataSampler(dataloader)

    # get 20 pic and draw them
    datas = []
    labels = []
    for i, (data, label) in enumerate(datasampler):
        if i == 20:
            break
        datas.append(transforms.ToPILImage()(data.squeeze(0)))
        labels.append(transforms.ToPILImage()(label.squeeze(0)))

    plt.figure(figsize=(20, 10))
    for i in range(10):
        plt.subplot(4, 10, i + 1)
        plt.imshow(datas[i])
        plt.subplot(4, 10, i + 11)
        plt.imshow(labels[i])

    for i in range(10):
        plt.subplot(4, 10, i + 21)
        plt.imshow(datas[i + 10])
        plt.subplot(4, 10, i + 31)
        plt.imshow(labels[i + 10])
    plt.show()


if __name__ == '__main__':
    test_HYPSO1_PNG_Dataset()
