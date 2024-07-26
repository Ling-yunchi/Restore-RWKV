import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.HYPSO1 import SameTransform


class REDATA(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms

        if mode == 'train':
            self.image_paths, self.mask_paths = self.get_image_mask_paths(
                os.path.join(root_dir, 'train', 'palsar_train'),
                os.path.join(root_dir, 'train', 'sentinel_train')
            )
        elif mode == 'test':
            self.sat_paths, self.gt_paths = self.get_image_mask_paths(
                os.path.join(root_dir, 'test', 'palsar'),
                os.path.join(root_dir, 'test', 'sentinel'),
                is_test=True
            )
        else:
            raise ValueError("Mode must be either 'train' or 'test'")

    @staticmethod
    def get_image_mask_paths(palsar_dir, sentinel_dir, is_test=False):
        image_paths = []
        mask_paths = []
        for subdir in ['palsar_train', 'sentinel_train'] if not is_test else ['palsar', 'sentinel']:
            img_dir = os.path.join(palsar_dir if 'palsar' in subdir else sentinel_dir,
                                   'image' if not is_test else 'sat')
            msk_dir = os.path.join(palsar_dir if 'palsar' in subdir else sentinel_dir, 'mask' if not is_test else 'gt')

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            msk_files = sorted([f for f in os.listdir(msk_dir) if f.endswith('.png')])

            for img, msk in zip(img_files, msk_files):
                image_paths.append(os.path.join(img_dir, img))
                mask_paths.append(os.path.join(msk_dir, msk))

        return image_paths, mask_paths

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_paths)
        elif self.mode == 'test':
            return len(self.sat_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = self.image_paths[idx]
            msk_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("L")
            mask = Image.open(msk_path).convert("L")
            if self.transforms:
                image, mask = self.transforms(image, mask)
            return image, mask
        elif self.mode == 'test':
            sat_path = self.sat_paths[idx]
            gt_path = self.gt_paths[idx]
            sat_image = Image.open(sat_path).convert("L")
            gt_image = Image.open(gt_path).convert("L")
            if self.transforms:
                sat_image, gt_image = self.transforms(sat_image, gt_image)
            return sat_image, gt_image


def is_gray(img, threshold=0.01):
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    transforms = SameTransform(transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]))
    train_dataset = REDATA("../dataset/REDATA", mode='train', transforms=transforms)
    test_dataset = REDATA("../dataset/REDATA", mode='test', transforms=transforms)

    print(len(train_dataset), len(test_dataset))
