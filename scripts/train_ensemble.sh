#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J partB_train
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:05
#BSUB -u s233498@student.dtu.dk
#BSUB -o job_logs/part_b_train_%J.out  # Output log file
#BSUB -e job_logs/part_b_train_%J.err  # Error log file

NUM_DECODERS=3
EPOCHS=5000
SEED=42

python -m venv temp_env
source temp_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python part_b_main.py train \
    --epochs $EPOCHS \
    --num-decoders $NUM_DECODERS \
    --seed $SEED