from typing import Callable, List
from torch.utils.data import Dataset
from torchvision import datasets


class FreeMatchUnlabeledDataset(Dataset):
    """
    Provides weak/strong augmented views for FreeMatch.
    Returns dict with both views so the trainer can apply the SAT/SAF logic.
    """

    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        indices: List[int],
        weak_transform: Callable,
        strong_transform: Callable,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        image, _ = self.base_dataset[dataset_idx]
        weak = self.weak_transform(image)
        strong = self.strong_transform(image)
        return {
            "weak": weak,
            "strong": strong,
            "dataset_idx": dataset_idx,
        }
    
