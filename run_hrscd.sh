#!/bin/bash
# Train TransSCD on the HRSCD dataset
# Make sure to run prepare_hrscd.py first to generate the cropped patches.

python train_SCD.py \
    --dataname "HRSCD" \
    --datapath "./datasets/HRSCD_256" \
    --num_classes 6 \
    --epoch 100 \
    --lr 3.5e-4 \
    --train_batchsize 2 \
    --val_batchsize 2 \
    --accum_steps 8
