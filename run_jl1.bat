@echo off
python train_SCD.py ^
    --dataname "JL1H" ^
    --datapath "path/to/JL1/train" ^
    --train_datapath "path/to/JL1/train" ^
    --val_datapath "path/to/JL1/test" ^
    --val_mode "test" ^
    --num_classes 6 ^
    --epoch 100 ^
    --train_batchsize 2 ^
    --val_batchsize 2 ^
    --accum_steps 8 ^
    --lr 3.5e-4
