# src/data/dataset.py
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset


class PanNukeDataset(Dataset):
    """Raw PanNuke loader. Returns numpy arrays, no transforms."""

    def __init__(self, split: str = "fold1"):
        self.data = load_dataset("RationAI/PanNuke", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = np.array(sample["image"], dtype=np.uint8)  # (256, 256, 3)

        mask = np.zeros((256, 256, 5), dtype=np.float32)
        for inst, cat in zip(sample["instances"], sample["categories"]):
            mask[:, :, cat] = np.maximum(mask[:, :, cat], np.array(inst, dtype=np.float32))

        return {"image": image, "mask": mask}