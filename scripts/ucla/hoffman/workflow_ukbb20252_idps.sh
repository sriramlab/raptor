#!/bin/bash

# Example workflow to embed volumes and then fit a downstream predictor.

# 1. Examine the manifest file and downstream labels file.
#  These must be provided by the user.

task_name=20252_wbu_inst2_idps_labels
manifest_file=manifests/20252_wbu_inst2_idps.txt
labels_file=data/${task_name}.csv

echo
echo "Manifest looks like this:"
head -n 5 $manifest_file
echo "..."
echo

echo "Labels file looks like this:"
head -5 $labels_file | cut -d, -f1-10
echo "..."
echo

# 2. Create and save random projection matrices

k10file=data/proj_normal_d1024_k10_run1
k100file=data/proj_normal_d1024_k100_run1

python create_projector.py --d 1024 --k 10 --saveas $k10file
python create_projector.py --d 1024 --k 100 --saveas $k100file

# 3. Embed the volumes

run_name=may19_DINO_ukbb20252
saveto=/u/scratch/u/ulzee/raptor/data/embs/$run_name

echo "Embeddings will be saved to: $saveto"
./scripts/ucla/hoffman/embed_ukbb.sh $saveto --k "$k10file,$k100file"

# 4. Train an MLP to predict the volumes

python -u fit_predictor.py \
    --embeddings $saveto/$k100file \
    --labels $labels_file \
    --regression --epochs 20

# 5. Measure accuracy

python scripts/ucla/score_ukbb20252.py \
    checkpoints/$task_name/predictions_test_${run_name}-${k100file}.csv
