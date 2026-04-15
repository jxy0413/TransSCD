@echo off
python test_SCD.py ^
    --dataname "JL1H" ^
    --datapath "path/to/JL1/test" ^
    --num_classes 6 ^
    --ckptpath "checkpoints/JL1H/TransSCD/run_0000/best.pth"
