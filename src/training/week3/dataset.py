
import torch

from torch.utils.data import Dataset


class SecureSenseDataset(Dataset):

    def __init__(self, data):

        self.input_ids = data["input_ids"]

        self.attention_mask = data["attention_mask"]

        self.labels = data["labels"]

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return {

            "input_ids": self.input_ids[idx],

            "attention_mask": self.attention_mask[idx],

            "labels": self.labels[idx]
        }
