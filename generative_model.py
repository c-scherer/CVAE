import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from abc import ABC, abstractmethod


class GenerativeModel(ABC, nn.Module):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, training_loader, n_epochs: int = 10):
        pass

    @abstractmethod
    def sample(self, n_samples: int):
        pass

# Define dataset
class CustomDataset(Dataset):
    def __init__(self, points, labels, transform=None, target_transform=None):
        self.points = points
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]
        if self.transform:
            points = self.transform(points)
        if self.target_transform:
            label = self.target_transform(label)
        return points, label