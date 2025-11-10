import argparse, json, logging, shlex, sys, tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor

from src.data import LabeledSubset, UnlabeledDataset
from src.models import DINOv3Classifier
from src.training.trainer import train

import mlflow
from .utils import MLflowTracker, setup_logging, flush_logging_handlers

from rich.traceback import install
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
    parser.add_argument("--experiment_name",                    type=str,   default="thesis")
    parser.add_argument("--run_name",                           type=str,   default=None)
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
    # ==================== EXPERIMENT TRACKING SETUP ===================
    # 1. PARSE ARGS & DIRECTORY SETUP
    args = parse_args()
    grouping_name = Path(args.json_grouping_path).stem
    model_name_safe = args.pretrained_name.replace("/", "-")
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    temp_dir_obj = tempfile.TemporaryDirectory()
    output_dir = Path(temp_dir_obj.name)
    # 2. LOGGING SETUP
    log_path = output_dir / f"{model_name_safe}.log"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    # 3. MLFLOW SETUP
    default_tracking_dir = Path("mlruns").resolve()
    default_tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(default_tracking_dir.as_uri())
    mlflow.set_experiment(args.experiment_name)
    # 4. RUN NAME
    run_name = args.run_name or f"{grouping_name}-{model_name_safe}"
    mlflow.start_run(run_name=run_name, tags={"experiment_version": args.experiment_version})
    # 5. INITIALIZE TRACKER
    tracker = MLflowTracker()
    tracker.log_params({k: v for k, v in vars(args).items()})
    tracker.log_text(" ".join(shlex.quote(arg) for arg in sys.argv), "command.txt")
    active_run = True
    # ===================================================================

    device = get_device(args.gpu_id)
    DINOv3Classifier.set_seed(args.seed)

    # Load grouping / splits
    grouping = load_grouping(args.json_grouping_path)
    tracker.log_dict(grouping, "grouping.json")
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

    logger.info("Device: %s", device)
    logger.info("Train (labeled) size: %d", len(train_subset))
    logger.info("Test size: %d", len(test_dataset))
    logger.info("Unlabeled size: %d", len(unlabeled_dataset))
    tracker.log_params(
        {
            "train_dataset_size": len(train_subset),
            "test_dataset_size": len(test_dataset),
            "unlabeled_dataset_size": len(unlabeled_dataset),
            "grouping_name": grouping_name,
        }
    )

    # Initialize model
    num_classes = len(train_dataset_base.classes)
    model = DINOv3Classifier(
        pretrained_name=args.pretrained_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
    )

    # Prepare output
    model_output_path = output_dir / f"{model_name_safe}.pth"

    metrics_path = model_output_path.with_suffix(".txt")
    try:
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
            tracker=tracker,
        )

        with open(metrics_path, "w") as f:
            f.write(f"Best train accuracy: {best_train_accuracy:.4f}\n")
            f.write(f"Best test accuracy: {best_test_accuracy:.4f}\n")

        logger.info("Training complete. Model saved to %s", model_output_path)
        logger.info("Metrics written to %s", metrics_path)

        flush_logging_handlers()
        if tracker:
            tracker.log_metrics(
                {
                    "best_train_accuracy": best_train_accuracy,
                    "best_test_accuracy": best_test_accuracy,
                }
            )
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
