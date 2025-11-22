#!/bin/bash
# Quick Start: Wan2.2 Animate on Multi-GPU
# Run this script to start the complete pipeline

echo "🚀 Starting Wan2.2 Animate Multi-GPU Pipeline"
echo ""
echo "This will:"
echo "  1. Download Wan2.2-Animate-14B model (~18.7 GB) to /home/caches"
echo "  2. Download preprocessing models (pose, face tracking)"
echo "  3. Preprocess your example video"
echo "  4. Generate animated video using 2x A40 GPUs"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Activate venv
source /workspace/Wan2.2/venv/bin/activate

# Run the pipeline
python /workspace/Wan2.2/run_wan_animate_multi_gpu.py \
    --num-gpus 2 \
    --refert-num 1 \
    --sample-steps 20 \
    --use-flux

echo ""
echo "✅ Complete! Check /workspace/Wan2.2/outputs/animate for results"
