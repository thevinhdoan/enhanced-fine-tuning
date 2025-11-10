#!/usr/bin/env bash

sudo -E ./.venv/bin/python3 -m experiments.run_experiment \
  --json_grouping_path src/grouping/cifar10--dinov3/train4_seed1312.json \
  --train_batch_size 2 \
  --test_batch_size 128 \
  --unlabeled_batch_size 128 \
  --unlabeled_sample_size_per_class 64 \
  --lambda_1 0.1 \
  --lambda_2 0.1 \
  --num_epochs 20 \
  --learning_rate 5e-4 \
  --gpu_id 0 \
  --experiment_version sanity \
  --unfreeze_last_n 1 \
  --seed 42
