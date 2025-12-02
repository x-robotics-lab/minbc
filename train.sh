#!/bin/bash
# MinBC Training Script
# Usage: bash train.sh

# Generate timestamp for unique experiment name
timestamp=$(date +%Y%m%d_%H%M%S)

# GPU Configuration
GPU_IDS="2,3"  # Modify this to your available GPUs
NUM_GPUS=2     # Should match the number of GPUs in GPU_IDS

# Training Hyperparameters
BATCH_SIZE=128
NUM_EPOCHS=100
LEARNING_RATE=0.0005
SEED=3

# Output configuration
OUTPUT_NAME="bc-${timestamp}"

# Set OpenMP threads to 1 for multi-GPU training stability
export OMP_NUM_THREADS=1

# Run training with torchrun for multi-GPU
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    train.py train \
    --gpu ${GPU_IDS} \
    --multi-gpu \
    --seed ${SEED} \
    --optim.num-epoch ${NUM_EPOCHS} \
    --optim.learning-rate ${LEARNING_RATE} \
    --optim.batch-size ${BATCH_SIZE} \
    --output_name "${OUTPUT_NAME}"

echo ""
echo "Training completed! Results saved to: outputs/${OUTPUT_NAME}"
echo "View logs with: tensorboard --logdir outputs/${OUTPUT_NAME}"
