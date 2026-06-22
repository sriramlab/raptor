#!/usr/bin/env bash
set -euo pipefail

# Optional preprocessing example: cache normalized, downsampled OCT volumes.
# The output .npz files contain an oct_volume_normalized array and can be embedded later.
# Edit these paths before running.
MANIFEST="data/oct_paths.txt"
CACHE_DIR="data/oct_cache_ds4"
ERRORS="data/oct_cache_errors.tsv"

python -u create_downsampled_oct_npz.py \
    --manifest "$MANIFEST" \
    --saveto "$CACHE_DIR" \
    --start 0 \
    --many 100 \
    --subsample_factor 4 \
    --skip_existing \
    --continue_on_error \
    --errors_log "$ERRORS"
