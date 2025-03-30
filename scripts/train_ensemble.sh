#!/bin/bash
#BSUB -J "ensemble[1-30]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 02:00
#BSUB -B                   
#BSUB -N
#BSUB -u s233498@dtu.dk
#BSUB -o job_logs/ensemble_%J_%I.out
#BSUB -e job_logs/ensemble_%J_%I.err

INDEX=${LSB_JOBINDEX}

if [ $INDEX -le 10 ]; then
    NUM_DECODERS=1
    COUNT=$INDEX
elif [ $INDEX -le 20 ]; then
    NUM_DECODERS=2
    COUNT=$(($INDEX - 10))
else
    NUM_DECODERS=3
    COUNT=$(($INDEX - 20))
fi

SEED=$COUNT

MODEL_NAME="ensemble_model_${NUM_DECODERS}_decoders_${COUNT}.pt"
EPOCHS=5000

echo "Job Index: $INDEX"
echo "NUM_DECODERS: $NUM_DECODERS"
echo "COUNT: $COUNT"
echo "SEED: $SEED"
echo "MODEL_NAME: $MODEL_NAME"
echo "EPOCHS: $EPOCHS"

python -m venv temp_env
source temp_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python part_b_main.py train \
    --epochs $EPOCHS \
    --num-decoders $NUM_DECODERS \
    --seed $SEED \
    --model-name $MODEL_NAME \
    --device cuda
