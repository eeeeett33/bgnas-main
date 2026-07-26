#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/autogllight/examples"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DEVICE="${DEVICE:-0}"

COMMON_ARGS=(
  --device "${DEVICE}"
  --epochs 150
  --hidden_dim 64
  --layer_number 4
  --dropout 0.5
  --model_lr 1e-3
  --model_wd 5e-4
  --arch_lr 1e-3
  --arch_wd 1e-3
  --gen_lr 1e-3
  --gen_wd 5e-4
  --grad_clip 5.0
  --trigger_size 3
  --target_class 0
  --vs_size 40
  --vs_number 40
  --selection_method cluster_degree
  --defense_mode prune
  --prune_thr 0
  --target_loss_weight 1.0
  --homo_loss_weight 100.0
  --retrain_epochs 200
  --retrain_lr 0.01
  --retrain_wd 5e-4
)

python darts_backdoor_minimal.py \
  "${COMMON_ARGS[@]}" \
  --dataset Pubmed --seed 1781  # 1333, 630

#python darts_backdoor_minimal.py \
#  "${COMMON_ARGS[@]}" \
#  --dataset Computers --seed 1649  # 763
#
#python darts_backdoor_minimal.py \
#  "${COMMON_ARGS[@]}" \
#  --dataset Photo --seed 1805  # 1465
#
#python darts_backdoor_minimal.py \
#  "${COMMON_ARGS[@]}" \
#  --dataset Flickr --seed 1012
#
#python darts_backdoor_minimal.py \
#  "${COMMON_ARGS[@]}" \
#  --dataset Cora --seed 1047  # 1330
#
#python darts_backdoor_minimal.py \
#  "${COMMON_ARGS[@]}" \
#  --dataset Citeseer --seed 1454
