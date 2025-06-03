#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate /u/project/sgss/UKBB/envs/medsam2

batch_size=128
if [ "$3" == "LlavaMed" ]; then
    batch_size=16
    conda activate /u/project/sgss/UKBB/envs/medsam
fi

nvidia-smi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

hostname=$(cat /proc/sys/kernel/hostname)
echo $hostname
# if [ "$hostname" == "g6" ]; then
#     export CUDA_VISIBLE_DEVICES=0
# fi

python -u $1 --folder $2 \
    --encoder $3 --manifest $4 --start $5 --many $6 --batch_size $batch_size \
    --saveto $7 $8 $9 ${10} ${11} ${12} ${13}

echo done
