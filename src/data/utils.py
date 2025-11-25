from typing import Callable, List
from torch.utils.data import Dataset
from torchvision import datasets


class LabeledSubset(Dataset):
    """Subset of a base dataset that also returns the original dataset index."""

    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        pixel_values, label = self.base_dataset[dataset_idx]
        return {
            "pixel_values": pixel_values,
            "label": label,
            "dataset_idx": dataset_idx,
        }


class AugmentedLabeledSubset(Dataset):
    """Labeled subset that applies a transform and returns the original index."""

    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int], transform: Callable):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        image, label = self.base_dataset[dataset_idx]
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "label": label,
            "dataset_idx": dataset_idx,
        }


class UnlabeledDataset(Dataset):
    """
    Wraps a base dataset and exposes only the pixel values for the provided indices.
    Items are dictionaries so that downstream DataLoader batches can be accessed
    via `batch["pixel_values"]`, which is what the trainer expects.
    """

    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        pixel_values, _ = self.base_dataset[dataset_idx]
        return {
            "pixel_values": pixel_values
        }
