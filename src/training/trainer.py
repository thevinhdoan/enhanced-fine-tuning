import gc
import logging
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.training.loss import Loss
from src.grouping.updater import recompute_grouping


def train(
    model,
    # data
    train_loader,
    test_loader, 
    unlabeled_set,
    # grouping info
    full_dataset_grouping,
    grouping_update_interval,
    # unlabeled data info
    unlabeled_sampling_method,
    unlabeled_indices,
    unlabeled_sample_size_per_cluster,
    unlabeled_sample_size_total,
    unlabeled_batch_size,
    # hyperparams for loss
    lambda_1,
    lambda_2,
    # other hyperparams
    device,
    num_epochs,
    learning_rate,
    num_workers,
    # output
    model_output_path,
    tracker
):

    # Setup
    logger = logging.getLogger(__name__)
    model.to(device)
    train_loss_fn = Loss(lambda_1=lambda_1, lambda_2=lambda_2)
    test_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_loss = float('inf')
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0

    base_dataset = getattr(getattr(train_loader, "dataset", None), "base_dataset", None)
    if grouping_update_interval and grouping_update_interval > 0 and base_dataset is None:
        raise ValueError("train_loader dataset must expose `base_dataset` when grouping updates are enabled")

    # Training
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        #################### TRAINING LOOP ####################
        train_loss = 0.0
        train_loss_standard = 0.0
        train_correct = 0
        for batch_train_idx, batch_train in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            batch_train_pixel_values = batch_train["pixel_values"]
            batch_train_labels = batch_train["label"]
            batch_train_indices = batch_train["dataset_idx"]
            batch_train_pixel_values = batch_train_pixel_values.to(device)
            batch_train_labels = batch_train_labels.to(device)

            # ===== METHOD 1: SAMPLE SOME UNLABELED DATA FROM EACH CLUSTER =====
            if unlabeled_sampling_method == 1:
                random.seed(42 + batch_train_idx + epoch)
                batch_grouping = {}
                batch_unlabeled_subsets = []
                # Go through each cluster in the training batch and sample data points
                for i, idx in enumerate(batch_train_indices):
                    member_ids = full_dataset_grouping[idx.item()]
                    sampled_ids = [unlabeled_indices.index(x) for x in random.sample(member_ids, k=min(unlabeled_sample_size_per_cluster, len(member_ids)))]
                    batch_unlabeled_subsets.append(Subset(unlabeled_set, sampled_ids))
                    batch_grouping[i] = list(range(batch_grouping.get(i-1, [-1])[-1] + 1, batch_grouping.get(i-1, [-1])[-1] + 1 + len(sampled_ids)))
                assert sum(len(v) for v in batch_grouping.values()) == sum(len(s) for s in batch_unlabeled_subsets)
                # Concatenate subsets from all clusters in the training batch
                batch_unlabeled_set = ConcatDataset(batch_unlabeled_subsets)
                batch_unlabeled_loader = DataLoader(
                    batch_unlabeled_set,
                    batch_size=unlabeled_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

            # ===== METHOD 2: DIRECTLY SAMPLE FROM THE UNLABELED SET =====
            elif unlabeled_sampling_method == 2:
                random.seed(42 + batch_train_idx + epoch)
                batch_unlabeled_ids = random.sample(unlabeled_indices, k=unlabeled_sample_size_total)
                # Create batch_grouping for the sampled unlabeled data
                # Negative keys are used for clusters not in the current training batch
                raw_grouping, batch_grouping = {}, {}
                for cluster_idx, member_ids in full_dataset_grouping.items():
                    raw_grouping[cluster_idx] = [i for i, x in enumerate(batch_unlabeled_ids) if x in member_ids]
                for i, idx in enumerate(batch_train_indices):
                    member_ids = raw_grouping[idx.item()]
                    batch_grouping[i] = member_ids
                for cluster_idx, member_ids in raw_grouping.items():
                    if cluster_idx not in batch_train_indices:
                        batch_grouping[-cluster_idx - 1] = member_ids
                assert sum(len(v) for v in batch_grouping.values()) == unlabeled_sample_size_total
                # Create dataset for the sampled unlabeled data
                sampled_ids = [unlabeled_indices.index(x) for x in batch_unlabeled_ids]
                batch_unlabeled_set = Subset(unlabeled_set, sampled_ids)
                batch_unlabeled_loader = DataLoader(
                    batch_unlabeled_set,
                    batch_size=unlabeled_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

            # ===== METHOD 3: DIRECTLY SAMPLE FROM THE UNLABELED SET  =====
            # =============== & MAKE SURE ALL THE CLUSTERS ARE THERE  =====
            # ... (to be implemented) ...
            else:
                raise ValueError(f"Unlabeled sampling method {unlabeled_sampling_method} not recognized.")

            batch_train_logits = model(pixel_values=batch_train_pixel_values)
            batch_unlabeled_logits = []
            for batch_unlabeled in batch_unlabeled_loader:
                batch_unlabeled_pixel_values = batch_unlabeled["pixel_values"]
                batch_unlabeled_pixel_values = batch_unlabeled_pixel_values.to(device)
                logits = model(pixel_values=batch_unlabeled_pixel_values)
                batch_unlabeled_logits.append(logits)
            batch_unlabeled_logits = torch.cat(batch_unlabeled_logits, dim=0)

            loss = train_loss_fn(batch_train_logits, batch_train_labels, batch_unlabeled_logits, batch_grouping)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (batch_train_logits.argmax(dim=1) == batch_train_labels).sum().item()

            loss = test_loss_fn(batch_train_logits, batch_train_labels)
            train_loss_standard += loss.item()
        
        del batch_train_pixel_values, batch_train_labels, batch_train_indices
        del batch_unlabeled_pixel_values, batch_unlabeled_logits
        gc.collect()
        torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_loss_standard /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        time_taken = time.time() - start_time
        logger.info("Epoch %d/%d - Train Loss: %.4f (%.4f) - Train Accuracy: %.4f - Time: %.2fs", 
                    epoch, num_epochs, train_loss, train_loss_standard, train_accuracy, time_taken)
        tracker.log_metrics(
            {
                "train_loss": train_loss,
                "train_loss_standard": train_loss_standard,
                "train_accuracy": train_accuracy,
                "train_epoch_time_sec": time_taken,
            },
            step=epoch,
        )
        
        #################### TESTING LOOP ################
        model.eval()
        test_loss = 0.0
        test_correct = 0
        start_time = time.time()
        with torch.no_grad():
            for batch_test in test_loader:
                batch_test_pixel_values, batch_test_labels = batch_test
                batch_test_pixel_values = batch_test_pixel_values.to(device)
                batch_test_labels = batch_test_labels.to(device)

                batch_test_logits = model(pixel_values=batch_test_pixel_values)
                loss = test_loss_fn(batch_test_logits, batch_test_labels)
                test_loss += loss.item()
                test_correct += (batch_test_logits.argmax(dim=1) == batch_test_labels).sum().item()

        del batch_test_pixel_values, batch_test_labels, batch_test_logits
        gc.collect()
        torch.cuda.empty_cache()

        test_loss /= len(test_loader)
        test_accuracy = test_correct / len(test_loader.dataset)
        time_taken = time.time() - start_time
        logger.info("Epoch %d/%d - Test Loss: %.4f - Test Accuracy: %.4f - Time: %.2fs",
                    epoch, num_epochs, test_loss, test_accuracy, time_taken)
        tracker.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_epoch_time_sec": time_taken,
            },
            step=epoch,
        )

        #################### CHECKPOINTING ################
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), model_output_path)
            logger.info("New best checkpoint saved to %s", model_output_path)
            tracker.log_metrics(
                {
                    "best_test_loss": best_test_loss,
                    "best_train_accuracy_so_far": best_train_accuracy,
                    "best_test_accuracy_so_far": best_test_accuracy,
                },
                step=epoch,
            )

        #################### GROUPING UPDATE ################
        if (
            grouping_update_interval
            and grouping_update_interval > 0
            and epoch % grouping_update_interval == 0
            and epoch < num_epochs
        ):
            effective_batch_size = 128
            full_dataset_grouping = recompute_grouping(
                model=model,
                base_dataset=base_dataset,
                train_indices=sorted(full_dataset_grouping.keys()),
                unlabeled_indices=unlabeled_indices,
                device=device,
                batch_size=effective_batch_size,
                num_workers=num_workers,
            )
            tracker.log_dict(full_dataset_grouping, f"grouping_epoch_{epoch}.json")

    gc.collect()
    torch.cuda.empty_cache()

    return best_train_accuracy, best_test_accuracy
