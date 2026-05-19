# src/data/preprocessing.py
"""Format transforms: augmentation, normalization, tensor conversion.

Wraps any dataset returning {"image": ndarray, "mask": ndarray}.
Augmentation is synchronized across image and mask.
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, raw_dataset, config: dict, is_train: bool = True, feature_fn=None):
        self.raw = raw_dataset
        self.feature_fn = feature_fn
        self.is_train = is_train

        norm = config.get("normalize", {})
        self.mean = np.array(norm.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.array(norm.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)

        aug = config.get("augmentation", {}) if is_train else {}
        self.h_flip = aug.get("horizontal_flip", False)
        self.v_flip = aug.get("vertical_flip", False)
        self.rotation = aug.get("rotation_degrees", 0)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        sample = self.raw[idx]
        if self.feature_fn:
            sample = self.feature_fn(sample)

        image = sample["image"].astype(np.float32) / 255.0
        mask = sample["mask"]

        if self.is_train:
            if self.h_flip and np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if self.v_flip and np.random.random() > 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if self.rotation:
                k = np.random.choice([0, 1, 2, 3])
                image = np.rot90(image, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask