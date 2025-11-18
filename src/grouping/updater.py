import logging
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import LabeledSubset


logger = logging.getLogger(__name__)


def _extract_backbone_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return backbone embeddings for a batch of pixel values."""
    outputs = model.backbone(pixel_values=pixel_values)
    feats = getattr(outputs, "pooler_output", None)
    if feats is None or feats.ndim != 2:
        feats = outputs.last_hidden_state[:, 0, :]
    return feats


def _compute_embeddings(
    model,
    base_dataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[int, np.ndarray]:
    """Compute embeddings for the requested dataset indices using the model backbone."""

    subset = LabeledSubset(base_dataset, indices)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    was_training = model.training
    model.eval()

    embeddings: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            dataset_indices = batch["dataset_idx"].tolist()
            feats = _extract_backbone_features(model, pixel_values)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            for dataset_idx, feat in zip(dataset_indices, feats):
                embeddings[int(dataset_idx)] = feat

    if was_training:
        model.train()

    return embeddings


def recompute_grouping(
    model,
    base_dataset,
    train_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[int, List[int]]:
    """
    Regenerate the grouping by assigning each unlabeled index to its closest labeled
    neighbor based on cosine similarity of backbone embeddings.
    """

    logger.info(
        "Recomputing grouping using %d labeled and %d unlabeled items",
        len(train_indices),
        len(unlabeled_indices),
    )

    # Compute embeddings for both labeled and unlabeled samples in a single pass.
    all_indices: List[int] = list(train_indices) + list(unlabeled_indices)
    embeddings = _compute_embeddings(
        model=model,
        base_dataset=base_dataset,
        indices=all_indices,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    def stack_embeddings(target_indices: Iterable[int]) -> np.ndarray:
        return np.stack([embeddings[int(idx)] for idx in target_indices]).astype(np.float32)

    train_embeddings = stack_embeddings(train_indices)
    unlabeled_embeddings = stack_embeddings(unlabeled_indices)

    # Normalize for cosine similarity and assign via nearest neighbor search.
    faiss.normalize_L2(train_embeddings)
    faiss.normalize_L2(unlabeled_embeddings)

    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_embeddings)
    _, nearest = index.search(unlabeled_embeddings, 1)

    grouping: Dict[int, List[int]] = {int(idx): [] for idx in train_indices}
    for unlabeled_idx, nearest_idx in zip(unlabeled_indices, nearest[:, 0]):
        labeled_idx = train_indices[int(nearest_idx)]
        grouping[int(labeled_idx)].append(int(unlabeled_idx))

    logger.info("Grouping recomputed")
    return grouping
