import json
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms


# ---- Paths ----
EMB_PATH = "src/grouping/cifar10--dinov3/embedding.pth"
OUT_DIR = Path("src/grouping/cifar10--dinov3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
SEEDS = [1312, 2201, 2411, 2503]
TRAIN_SAMPLE_SIZES_PER_CLASS = [4, 8, 16, 25, 32, 40, 64, 400]  # k per class
NUM_CLASSES = 10


def load_labels():
    """
    Load CIFAR-10 train set to get labels in the exact canonical order (length 50000).
    """
    dataset = CIFAR10(
        root="datasets",
        train=True,
        download=True,
        transform=transforms.ToTensor()    # transform doesn't matter, we only need labels
    )
    labels = torch.tensor(dataset.targets, dtype=torch.long)   # shape (50000,)
    return labels


def sample_indices_per_class(labels, k, seed):
    """
    Sample *k* items per class using the canonical dataset index ordering.
    """
    rng = random.Random(seed)
    class_to_indices = {c: [] for c in range(NUM_CLASSES)}

    for idx, c in enumerate(labels.tolist()):
        class_to_indices[c].append(idx)

    # sample
    train_indices = []
    for c in range(NUM_CLASSES):
        train_indices.extend(rng.sample(class_to_indices[c], k))

    return sorted(train_indices)


def main():
    embeddings = torch.load(EMB_PATH)   # shape (50000, D)
    embeddings_np = embeddings.cpu().numpy().astype(np.float32)
    labels = load_labels()              # shape (50000,)

    for k in TRAIN_SAMPLE_SIZES_PER_CLASS:
        for seed in SEEDS:

            # 1) pick k per class
            train_indices = sample_indices_per_class(labels, k, seed)
            train_set = set(train_indices)
            unlabeled_indices = [i for i in range(len(labels)) if i not in train_set]

            # 2) normalize for cosine similarity via inner product
            train_embeddings = embeddings_np[train_indices]
            unlabeled_embeddings = embeddings_np[unlabeled_indices]
            train_embeddings /= np.linalg.norm(train_embeddings, axis=1, keepdims=True)
            unlabeled_embeddings /= np.linalg.norm(unlabeled_embeddings, axis=1, keepdims=True)

            d = embeddings_np.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(train_embeddings)

            # 3) assign each unlabeled item to nearest labeled item
            _, nearest = index.search(unlabeled_embeddings, 1)

            grouping = defaultdict(list)
            for i, unl_idx in enumerate(unlabeled_indices):
                labeled_local = int(nearest[i, 0])
                labeled_global = train_indices[labeled_local]
                grouping[labeled_global].append(unl_idx)

            # ensure all are present
            for t in train_indices:
                grouping[t]  # touch to keep empty groups

            # sort for stability
            grouping_sorted = {k2: sorted(v2) for k2, v2 in sorted(grouping.items())}

            # 4) write output
            out_path = OUT_DIR / f"train{k}_seed{seed}.json"
            with out_path.open("w") as f:
                json.dump(grouping_sorted, f, indent=4)

            print(f"[k={k}, seed={seed}] wrote {out_path} "
                  f"(labeled={len(train_indices)}, unlabeled={len(unlabeled_indices)})")


if __name__ == "__main__":
    main()
