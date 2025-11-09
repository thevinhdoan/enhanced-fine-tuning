import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor

from rich.traceback import install

from src.data import LabeledSubset, UnlabeledDataset
from src.models import DINOv3Classifier
from src.training.trainer import train


install()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_grouping_path",                 type=str,   required=True)
    parser.add_argument("--train_batch_size",                   type=int,   default=64)
    parser.add_argument("--test_batch_size",                    type=int,   default=128)
    parser.add_argument("--unlabeled_batch_size",               type=int,   default=256)
    parser.add_argument("--unlabeled_sample_size_per_class",    type=int,   default=32)
    parser.add_argument("--lambda_1",                           type=float, default=1.0)
    parser.add_argument("--lambda_2",                           type=float, default=1.0)
    parser.add_argument("--num_epochs",                         type=int,   default=10)
    parser.add_argument("--learning_rate",                      type=float, default=5e-4)
    parser.add_argument("--num_workers",                        type=int,   default=4)
    parser.add_argument("--gpu_id",                             type=int,   default=0)
    parser.add_argument("--experiment_version",                 type=str,   default="v1")
    parser.add_argument("--pretrained_name",                    type=str,   default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--freeze_backbone",                    action="store_true")
    parser.add_argument("--unfreeze_last_n",                    type=int,   default=0)
    parser.add_argument("--seed",                               type=int,   default=42)
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def load_grouping(grouping_path: str) -> Dict[int, List[int]]:
    with open(grouping_path, "r") as f:
        grouping_raw = json.load(f)
    grouping = {int(k): sorted(int(idx) for idx in v) for k, v in grouping_raw.items()}
    return dict(sorted(grouping.items(), key=lambda item: item[0]))


def get_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def main():
    args = parse_args()

    device = get_device(args.gpu_id)
    DINOv3Classifier.set_seed(args.seed)

    # Load grouping / splits
    grouping = load_grouping(args.json_grouping_path)
    train_indices = sorted(grouping.keys())
    unlabeled_indices = sorted({idx for members in grouping.values() for idx in members})

    overlap = set(train_indices).intersection(unlabeled_indices)
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping indices between labeled and unlabeled sets: {sorted(list(overlap))[:5]}...")

    # Data
    processor = AutoImageProcessor.from_pretrained(args.pretrained_name)
    transform = lambda image: processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    train_dataset_base = datasets.CIFAR10(root="datasets/", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="datasets/", train=False, download=True, transform=transform)

    # Labeled train subset with indices propagated to the trainer
    train_subset = LabeledSubset(train_dataset_base, train_indices)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    unlabeled_dataset = UnlabeledDataset(train_dataset_base, unlabeled_indices)

    print(f"Device: {device}")
    print(f"Train (labeled) size: {len(train_subset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Unlabeled size: {len(unlabeled_dataset)}")

    # Initialize model
    num_classes = len(train_dataset_base.classes)
    model = DINOv3Classifier(
        pretrained_name=args.pretrained_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
    )

    # Prepare output
    grouping_name = Path(args.json_grouping_path).stem
    model_name_safe = args.pretrained_name.replace("/", "-")
    output_dir = Path("outputs") / args.experiment_version / grouping_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = output_dir / f"{model_name_safe}.pth"

    # Training
    best_train_accuracy, best_test_accuracy = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        unlabeled_set=unlabeled_dataset,
        full_dataset_grouping=grouping,
        unlabeled_indices=unlabeled_indices,
        unlabeled_sample_size_per_class=args.unlabeled_sample_size_per_class,
        unlabeled_batch_size=args.unlabeled_batch_size,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        model_output_path=str(model_output_path),
    )

    metrics_path = model_output_path.with_suffix(".txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best train accuracy: {best_train_accuracy:.4f}\n")
        f.write(f"Best test accuracy: {best_test_accuracy:.4f}\n")

    print(f"Training complete. Model saved to {model_output_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
