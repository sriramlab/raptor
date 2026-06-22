#!/usr/bin/env bash
set -euo pipefail

# Example: embed a MedMNIST-style 3D npz with DINO/DINOv2.
# Edit these paths before running.
NPZ_PATH="data/nodulemnist3d_64.npz"
MANIFEST="data/nodulemnist3d_64.txt"
PROJECTOR="data/proj_normal_d1024_k100_run1.npy"
OUTDIR="data/embs/nodule_DINO"

python create_projector.py --seed 0 --d 1024 --k 100 --saveas "${PROJECTOR%.npy}"

python -u embed.py --npz "$NPZ_PATH" \
    --encoder DINO \
    --manifest "$MANIFEST" \
    --start 0 --many 100 \
    --batch_size 128 \
    --saveto "$OUTDIR" \
    --k "$PROJECTOR"
