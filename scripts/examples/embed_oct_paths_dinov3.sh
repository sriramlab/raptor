#!/usr/bin/env bash
set -euo pipefail

# Example: embed OCT volumes with DINOv3.
# MANIFEST should contain absolute paths to OCT image folders, OCT .npz files, or OCT .zip files.
# Edit these paths before running.
MANIFEST="data/oct_paths.txt"
PROJECTOR="data/proj_normal_d4096_k100_run1.npy"
OUTDIR="data/embs/oct_DINOv3_globalpad"
ERRORS="data/oct_errors.tsv"

python create_projector.py --seed 0 --d 4096 --k 100 --saveas "${PROJECTOR%.npy}"

python -u embed.py --oct_path_list \
    --encoder DINOv3 \
    --manifest "$MANIFEST" \
    --start 0 --many 100 \
    --batch_size 8 \
    --planes ACS \
    --subsample_factor 4 \
    --resize_mode global_pad \
    --skip_existing \
    --continue_on_error \
    --errors_log "$ERRORS" \
    --saveto "$OUTDIR" \
    --k "$PROJECTOR"
