@echo off
python train_SCD.py ^
    --dataname "Landsat" ^
    --datapath "path/to/Landsat-SCD" ^
    --num_classes 5 ^
    --epoch 100 ^
    --train_batchsize 2 ^
    --val_batchsize 2 ^
    --accum_steps 8 ^
    --lr 3.5e-4
