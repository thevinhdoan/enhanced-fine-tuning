import argparse
import json
import logging
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

import mlflow

from src.data import AugmentedLabeledSubset, FreeMatchUnlabeledDataset
from src.models import DINOv3Classifier
from .utils import MLflowTracker, setup_logging, flush_logging_handlers

from rich.traceback import install

install()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_grouping_path", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--experiment_version", type=str, default="freematch_v1")
    parser.add_argument(
        "--pretrained_name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_last_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="thesis")
    parser.add_argument("--run_name", type=str, default=None)

    # FreeMatch specific
    parser.add_argument("--unlabeled_ratio", type=int, default=7, help="mu in the paper")
    parser.add_argument("--unsup_weight", type=float, default=1.0, help="wu in Eq. 12")
    parser.add_argument("--fairness_weight", type=float, default=0.5, help="wf in Eq. 12")
    parser.add_argument("--ema_momentum", type=float, default=0.999, help="lambda for EMA stats")
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


def build_transforms(processor: AutoImageProcessor):
    processor_only = lambda image: processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    weak_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            processor_only,
        ]
    )
    strong_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            processor_only,
        ]
    )
    test_transform = processor_only
    return weak_transform, strong_transform, test_transform


def normalize_distribution(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    total = x.sum()
    if total <= eps:
        return torch.full_like(x, 1.0 / x.numel())
    return x / total


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    unlabeled_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    epoch: int,
    num_epochs: int,
    unsup_weight: float,
    fairness_weight: float,
    ema_momentum: float,
    tau_global: torch.Tensor,
    p_model_ema: torch.Tensor,
    hist_ema: torch.Tensor,
    tracker: MLflowTracker,
) -> Dict[str, float]:
    logger = logging.getLogger(__name__)
    ce_loss = torch.nn.CrossEntropyLoss()
    model.train()
    unlabeled_iter = iter(unlabeled_loader)

    train_sup_loss = 0.0
    train_unsup_loss = 0.0
    train_fairness_loss = 0.0
    train_correct = 0
    num_batches = 0

    for batch_idx, batch_labeled in enumerate(train_loader):
        try:
            batch_unlabeled = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            batch_unlabeled = next(unlabeled_iter)

        batch_labeled_pixels = batch_labeled["pixel_values"].to(device)
        batch_labels = batch_labeled["label"].to(device)
        weak_pixels = batch_unlabeled["weak"].to(device)
        strong_pixels = batch_unlabeled["strong"].to(device)

        logits_labeled = model(pixel_values=batch_labeled_pixels)
        sup_loss = ce_loss(logits_labeled, batch_labels)

        with torch.no_grad():
            logits_weak = model(pixel_values=weak_pixels)
            probs_weak = torch.softmax(logits_weak, dim=1)
            max_probs, pseudo_labels = torch.max(probs_weak, dim=1)

            batch_conf_mean = max_probs.mean()
            tau_global = ema_momentum * tau_global + (1 - ema_momentum) * batch_conf_mean
            p_model_ema = ema_momentum * p_model_ema + (1 - ema_momentum) * probs_weak.mean(dim=0)

            weak_hist = torch.bincount(pseudo_labels, minlength=num_classes).float()
            weak_hist = normalize_distribution(weak_hist)
            hist_ema = ema_momentum * hist_ema + (1 - ema_momentum) * weak_hist

            tau_per_class = (p_model_ema / p_model_ema.max().clamp(min=1e-6)) * tau_global

        logits_strong = model(pixel_values=strong_pixels)
        per_sample_unsup = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")

        mask = (max_probs >= tau_per_class[pseudo_labels]).float()
        if mask.sum() > 0:
            unsup_loss = (per_sample_unsup * mask).sum() / mask.sum()
        else:
            unsup_loss = torch.tensor(0.0, device=device)

        probs_strong = torch.softmax(logits_strong, dim=1)
        pseudo_strong = torch.argmax(probs_strong, dim=1)
        masked_probs = probs_strong * mask.unsqueeze(1)
        if mask.sum() > 0:
            p_batch = masked_probs.sum(dim=0) / mask.sum()
            hist_batch = torch.zeros(num_classes, device=device)
            hist_batch.scatter_add_(0, pseudo_strong, mask)
            hist_batch = normalize_distribution(hist_batch)
        else:
            p_batch = torch.full((num_classes,), 1.0 / num_classes, device=device)
            hist_batch = torch.full((num_classes,), 1.0 / num_classes, device=device)

        target_dist = normalize_distribution(hist_ema * p_model_ema)
        batch_dist = normalize_distribution(hist_batch * p_batch)
        fairness_loss = -torch.sum(target_dist * torch.log(batch_dist + 1e-8))

        total_loss = sup_loss + unsup_weight * unsup_loss + fairness_weight * fairness_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_sup_loss += sup_loss.item()
        train_unsup_loss += unsup_loss.item()
        train_fairness_loss += fairness_loss.item()
        train_correct += (logits_labeled.argmax(dim=1) == batch_labels).sum().item()
        num_batches += 1

    train_sup_loss /= max(1, num_batches)
    train_unsup_loss /= max(1, num_batches)
    train_fairness_loss /= max(1, num_batches)
    train_accuracy = train_correct / len(train_loader.dataset)

    logger.info(
        "Epoch %d/%d - Train sup: %.4f | unsup: %.4f | fairness: %.4f | acc: %.4f",
        epoch,
        num_epochs,
        train_sup_loss,
        train_unsup_loss,
        train_fairness_loss,
        train_accuracy,
    )

    tracker.log_metrics(
        {
            "train_sup_loss": train_sup_loss,
            "train_unsup_loss": train_unsup_loss,
            "train_fairness_loss": train_fairness_loss,
            "train_accuracy": train_accuracy,
        },
        step=epoch,
    )

    test_loss = 0.0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for pixels, labels in test_loader:
            pixels = pixels.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixels)
            loss = ce_loss(logits, labels)
            test_loss += loss.item()
            test_correct += (logits.argmax(dim=1) == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)
    logger.info(
        "Epoch %d/%d - Test loss: %.4f | Test acc: %.4f",
        epoch,
        num_epochs,
        test_loss,
        test_accuracy,
    )
    tracker.log_metrics(
        {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        step=epoch,
    )

    return {
        "train_sup_loss": train_sup_loss,
        "train_unsup_loss": train_unsup_loss,
        "train_fairness_loss": train_fairness_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "tau_global": tau_global.detach(),
        "p_model_ema": p_model_ema.detach(),
        "hist_ema": hist_ema.detach(),
    }


def main():
    args = parse_args()
    grouping_name = Path(args.json_grouping_path).stem
    model_name_safe = args.pretrained_name.replace("/", "-")
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    temp_dir_obj = tempfile.TemporaryDirectory()
    output_dir = Path(temp_dir_obj.name)

    log_path = output_dir / f"{model_name_safe}.log"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    default_tracking_dir = Path("mlruns").resolve()
    default_tracking_dir.mkdir(parents=True, exist_ok=True)
    if os.getenv("MLFLOW_TRACKING_URI", None):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        mlflow.set_tracking_uri(default_tracking_dir.as_uri())
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"{grouping_name}-{model_name_safe}-freematch"
    mlflow.start_run(run_name=run_name, tags={"experiment_version": args.experiment_version})
    tracker = MLflowTracker()
    tracker.log_params({k: v for k, v in vars(args).items()})
    tracker.log_text(" ".join(shlex.quote(arg) for arg in sys.argv), "command.txt")
    active_run = True

    device = get_device(args.gpu_id)
    DINOv3Classifier.set_seed(args.seed)

    grouping = load_grouping(args.json_grouping_path)
    tracker.log_dict(grouping, "grouping.json")
    train_indices = sorted(grouping.keys())
    unlabeled_indices = sorted({idx for members in grouping.values() for idx in members})

    overlap = set(train_indices).intersection(unlabeled_indices)
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} overlapping indices between labeled and unlabeled sets: {sorted(list(overlap))[:5]}..."
        )

    processor = AutoImageProcessor.from_pretrained(args.pretrained_name)
    weak_transform, strong_transform, test_transform = build_transforms(processor)

    base_train = datasets.CIFAR10(root="datasets/", train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root="datasets/", train=False, download=True, transform=test_transform)

    train_dataset = AugmentedLabeledSubset(base_train, train_indices, transform=weak_transform)
    unlabeled_dataset = FreeMatchUnlabeledDataset(
        base_train,
        unlabeled_indices,
        weak_transform=weak_transform,
        strong_transform=strong_transform,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.train_batch_size * args.unlabeled_ratio,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    logger.info("Device: %s", device)
    logger.info("Train (labeled) size: %d", len(train_dataset))
    logger.info("Test size: %d", len(test_dataset))
    logger.info("Unlabeled size: %d", len(unlabeled_dataset))
    tracker.log_params(
        {
            "train_dataset_size": len(train_dataset),
            "test_dataset_size": len(test_dataset),
            "unlabeled_dataset_size": len(unlabeled_dataset),
            "grouping_name": grouping_name,
        }
    )

    num_classes = len(base_train.classes)
    model = DINOv3Classifier(
        pretrained_name=args.pretrained_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
    )
    model.to(device)

    model_output_path = output_dir / f"{model_name_safe}_freematch.pth"
    metrics_path = model_output_path.with_suffix(".txt")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tau_global = torch.tensor(1.0 / num_classes, device=device)
    p_model_ema = torch.full((num_classes,), 1.0 / num_classes, device=device)
    hist_ema = torch.full((num_classes,), 1.0 / num_classes, device=device)

    best_test_accuracy = 0.0
    best_train_accuracy = 0.0
    best_test_loss = float("inf")

    try:
        for epoch in range(1, args.num_epochs + 1):
            metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                unlabeled_loader=unlabeled_loader,
                test_loader=test_loader,
                device=device,
                optimizer=optimizer,
                num_classes=num_classes,
                epoch=epoch,
                num_epochs=args.num_epochs,
                unsup_weight=args.unsup_weight,
                fairness_weight=args.fairness_weight,
                ema_momentum=args.ema_momentum,
                tau_global=tau_global,
                p_model_ema=p_model_ema,
                hist_ema=hist_ema,
                tracker=tracker,
            )

            tau_global = metrics["tau_global"]
            p_model_ema = metrics["p_model_ema"]
            hist_ema = metrics["hist_ema"]

            if metrics["test_loss"] < best_test_loss:
                best_test_loss = metrics["test_loss"]
                best_test_accuracy = metrics["test_accuracy"]
                best_train_accuracy = metrics["train_accuracy"]
                torch.save(model.state_dict(), model_output_path)
                logger.info("New best checkpoint saved to %s", model_output_path)
                tracker.log_metrics(
                    {
                        "best_test_loss": best_test_loss,
                        "best_test_accuracy": best_test_accuracy,
                        "best_train_accuracy": best_train_accuracy,
                    },
                    step=epoch,
                )

        with open(metrics_path, "w") as f:
            f.write(f"Best train accuracy: {best_train_accuracy:.4f}\n")
            f.write(f"Best test accuracy: {best_test_accuracy:.4f}\n")

        logger.info("Training complete. Model saved to %s", model_output_path)
        logger.info("Metrics written to %s", metrics_path)

        flush_logging_handlers()
        tracker.log_artifact(model_output_path)
        tracker.log_artifact(metrics_path)
        tracker.log_artifact(log_path)
    finally:
        if active_run:
            mlflow.end_run()
            active_run = False
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
