#!/bin/bash
#BSUB -J "cov"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 02:00
#BSUB -B                   
#BSUB -N
#BSUB -u s243867@dtu.dk
#BSUB -o job_logs/cov_%J_%I.out
#BSUB -e job_logs/cov_%J_%I.err

# Set shared arguments
EXPERIMENT_FOLDER="ensemble_experiment"
DEVICE="cuda"
NUM_PAIRS=10
EPOCHS_PER_GEODESIC=50
NUM_T=20
LATENT_DIM=2

# Loop over decoder counts
for NUM_DECODERS in 1 2 3
do
    # Run 10 times for each decoder count
    for RUN_ID in {1..10}
    do
        MODEL_NAME="ensemble_model_${NUM_DECODERS}_decoder"
        # Add 's' if more than 1 decoder for nice naming
        if [ "$NUM_DECODERS" -gt 1 ]; then
            MODEL_NAME="${MODEL_NAME}s"
        fi
        MODEL_NAME="${MODEL_NAME}_${RUN_ID}.pt"

        echo "Running: ${MODEL_NAME}"

        python part_b_main.py geodesic \
            --experiment-folder "$EXPERIMENT_FOLDER" \
            --model-name "$MODEL_NAME" \
            --device "$DEVICE" \
            --num-pairs "$NUM_PAIRS" \
            --epochs-per-geodesic "$EPOCHS_PER_GEODESIC" \
            --num-t "$NUM_T" \
            --num-decoders "$NUM_DECODERS" \
            --latent-dim "$LATENT_DIM"
    done
done