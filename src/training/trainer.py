import gc
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.training.loss import Loss


def train(
    model,
    # data
    train_loader,
    test_loader, 
    unlabeled_set,
    # grouping info
    full_dataset_grouping,
    # unlabeled data info
    unlabeled_indices,
    unlabeled_sample_size_per_class,
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
    model_output_path
):

    # Setup
    model.to(device)
    train_loss_fn = Loss(lambda_1=lambda_1, lambda_2=lambda_2)
    test_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_loss = float('inf')
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0

    # Training
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        #################### TRAINING LOOP ####################
        train_loss = 0.0
        train_loss_standard = 0.0
        train_correct = 0
        for batch_train_idx, batch_train in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            # batch_train_pixel_values, batch_train_labels, batch_train_indices = batch_train
            batch_train_pixel_values = batch_train["pixel_values"]
            batch_train_labels = batch_train["label"]
            batch_train_indices = batch_train["dataset_idx"]
            batch_train_pixel_values = batch_train_pixel_values.to(device)
            batch_train_labels = batch_train_labels.to(device)

            # ===== METHOD 1: SAMPLE SOME UNLABELED DATA FROM EACH CLUSTER =====
            random.seed(42 + batch_train_idx + epoch)
            batch_grouping = {}
            batch_unlabeled_subsets = []
            for i, idx in enumerate(batch_train_indices):
                member_ids = full_dataset_grouping[idx.item()]
                sampled_ids = [unlabeled_indices.index(x) for x in random.choices(member_ids, k=min(unlabeled_sample_size_per_class, len(member_ids)))]
                batch_unlabeled_subsets.append(Subset(unlabeled_set, sampled_ids))
                batch_grouping[i] = list(range(batch_grouping.get(i-1, [-1])[-1] + 1, batch_grouping.get(i-1, [-1])[-1] + 1 + len(sampled_ids)))
            batch_unlabeled_set = ConcatDataset(batch_unlabeled_subsets)
            batch_unlabeled_loader = DataLoader(
                batch_unlabeled_set,
                batch_size=unlabeled_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            # =================================================================

            # ===== METHOD 2: DIRECTLY SAMPLE FROM THE UNLABELED SET =====
            # ... (to be implemented) ...
            # ============================================================

            # ===== METHOD 3: DIRECTLY SAMPLE FROM THE UNLABELED SET  =====
            # =============== & MAKE SURE ALL THE CLUSTERS ARE THERE  =====
            # ... (to be implemented) ...
            # ============================================================

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
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} ({train_loss_standard:.4f}) - Train Accuracy: {train_accuracy:.4f} - Time: {time_taken:.2f}s")
        
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
        print(f"Epoch {epoch}/{num_epochs} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - Time: {time_taken:.2f}s")

        #################### CHECKPOINTING ################
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), model_output_path)
            print("New best checkpoint saved.")
    
    gc.collect()
    torch.cuda.empty_cache()

    return best_train_accuracy, best_test_accuracy
