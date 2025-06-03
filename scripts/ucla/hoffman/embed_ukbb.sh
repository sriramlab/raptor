#!/bin/bash

saveto=$1

ukbbroot=/u/project/sgss/UKBB/imaging/bulk
dset=20252
GPU=RTX2080Ti
EMB_SCRIPT="embed.py"
manifest=data/20252_wbu_inst2_idps.txt
encoder=DINO

data_folder=$ukbbroot/$dset

mkdir $saveto
lns=$(wc -l $manifest | awk ' { print $1 }')
echo $lns
if [ $lns -eq 0 ]; then
    continue
fi
lns=$((lns + 1)) # handle some edge cases with total count

runtime="10:00:00"
if [[ $manifest == *"missing"* ]]; then
    bsize=$lns
    runtime="00:15:00"
else
    bsize=$((lns / 16))
    if [ $bsize -eq 0 ]; then
        bsize=1
    fi
fi

echo $lns $bsize

for (( i=0; i<lns; i+=bsize ))
do
    qsub -cwd -N ${GPU}-${encoder}-${dset} -l $GPU,gpu,cuda=1,h_rt=$runtime -o logs -j y \
        ./scripts/ucla/hoffman/job_embed.sh $EMB_SCRIPT $data_folder $encoder $manifest $i $bsize $saveto $2 $3 $4 $5 $6 $7
    # break
done