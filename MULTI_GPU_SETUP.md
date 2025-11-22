# Wan2.2 Animate Multi-GPU Setup Guide

This guide will help you run Wan2.2-Animate-14B on your multi-GPU setup (2x A40 48GB).

## Overview

Based on research of the Wan2.2 codebase and GitHub issues, here's what you need to know:

### Key Findings from GitHub Issues Research:

1. **Multi-GPU is Essential for 14B Models**: Single GPU with <80GB VRAM cannot run the 14B models. Your 2x A40 (48GB each = 96GB total) is suitable.

2. **GGUF Model Limitation**: The GGUF quantized models from QuantStack are **not directly compatible** with the official Wan2.2 codebase. The codebase uses safetensors format with FSDP (Fully Sharded Data Parallel).

3. **Common Issues**:
   - OOM errors even on A6000 (48GB) - solved by using multi-GPU with FSDP
   - Missing preprocessing dependencies (librosa, decord, peft, onnxruntime)
   - CUDA allocation errors - solved with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## System Requirements

- **Hardware**: 2x NVIDIA A40 (48GB VRAM each)
- **Software**: 
  - Python 3.10+
  - PyTorch 2.4+ with CUDA 12.1+
  - CUDA-compatible GPU drivers

## Quick Start

### 1. Activate Virtual Environment

```bash
cd /workspace/Wan2.2
source venv/bin/activate
```

### 2. Install Additional Dependencies

```bash
# Install animate-specific requirements
pip install -r requirements_animate.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### 3. Run the Setup Script

```bash
# Make scripts executable
chmod +x setup_wan_animate.sh
chmod +x run_wan_animate_multi_gpu.py

# Run the complete pipeline
./setup_wan_animate.sh
```

## What the Scripts Do

### `run_wan_animate_multi_gpu.py`

This Python script:
1. **Downloads Models** (to `/home/caches`):
   - Wan2.2-Animate-14B main model (~18.7 GB in safetensors format)
   - Preprocessing models: YOLOv10, VitPose, SAM2
   
2. **Preprocesses Video**:
   - Extracts pose information from driving video
   - Generates face tracking data
   - Applies pose retargeting (optional FLUX enhancement)
   - Outputs: `src_pose.mp4`, `src_face.mp4`, `src_ref.png`

3. **Runs Multi-GPU Inference**:
   - Uses FSDP (Fully Sharded Data Parallel)
   - DeepSpeed Ulysses sequence parallelism
   - Distributes model across 2 GPUs
   - Generates animated video

### Command Line Options

```bash
python run_wan_animate_multi_gpu.py --help

Options:
  --skip-download         Skip model download if already exists
  --skip-preprocessing    Skip video preprocessing
  --video PATH           Path to driving video (default: examples/wan_animate/animate/video.mp4)
  --image PATH           Path to reference image (default: examples/wan_animate/animate/image.jpeg)
  --num-gpus N           Number of GPUs (default: 2)
  --refert-num {1,5}     Temporal guidance frames (default: 1)
  --sample-steps N       Diffusion steps (default: 20, more=better quality but slower)
  --use-flux             Enhanced pose retargeting with FLUX (recommended)
  --replace-mode         Use character replacement instead of animation
  --use-relighting-lora  Use relighting LoRA for replacement mode
```

## Step-by-Step Manual Execution

If you prefer to run steps individually:

### Step 1: Download Models

```bash
python run_wan_animate_multi_gpu.py --skip-preprocessing
```

This downloads all required models to `/home/caches/`.

### Step 2: Preprocess Your Video

```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path /workspace/Wan2.2/examples/wan_animate/animate/video.mp4 \
    --refer_path /workspace/Wan2.2/examples/wan_animate/animate/image.jpeg \
    --save_path /workspace/Wan2.2/outputs/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag \
    --use_flux
```

**Important Parameters**:
- `--resolution_area`: Target resolution (width height in pixels)
- `--retarget_flag`: Enable pose retargeting (recommended for different body proportions)
- `--use_flux`: Use FLUX for enhanced retargeting (better quality)

### Step 3: Run Multi-GPU Inference

```bash
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 2 \
    generate.py \
    --task animate-14B \
    --ckpt_dir /home/caches/Wan2.2-Animate-14B \
    --src_root_path /workspace/Wan2.2/outputs/animate/process_results \
    --refert_num 1 \
    --sample_steps 20 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2
```

**Key Flags**:
- `--dit_fsdp`: Enable FSDP for DiT model (distributes model across GPUs)
- `--t5_fsdp`: Enable FSDP for T5 text encoder
- `--ulysses_size 2`: Sequence parallelism across 2 GPUs
- `--refert_num 1`: Use 1 frame for temporal guidance (faster, use 5 for better quality)

## Performance Expectations

Based on the computational efficiency table in the README:

| Model | GPUs | Resolution | Time | Memory/GPU |
|-------|------|------------|------|------------|
| Animate-14B | 2x A40 | 720p | ~30-40 min | ~35-40 GB |
| Animate-14B | 1x A40 | 720p | OOM Error | N/A |

**Tips for Optimization**:
- Reduce `--sample-steps` (20 is good, 15 for faster but lower quality)
- Use `--refert_num 1` instead of 5 for faster generation
- Lower resolution: `--resolution_area 960 544` instead of `1280 720`

## Troubleshooting

### 1. CUDA Out of Memory (OOM)

```
torch.OutOfMemoryError: CUDA out of memory
```

**Solutions**:
- Ensure both GPUs are being used: `nvidia-smi` should show processes on both
- Try: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Reduce batch size or resolution
- Close other GPU processes

### 2. Missing Dependencies

```
ModuleNotFoundError: No module named 'librosa'
```

**Solution**:
```bash
pip install -r requirements_animate.txt
pip install librosa decord peft onnxruntime-gpu
```

### 3. Preprocessing Fails

```
ERROR: Pose detection failed
```

**Solutions**:
- Ensure preprocessing models are downloaded
- Check video quality and format (MP4 works best)
- Verify reference image is clear and shows full body

### 4. NCCL/Distributed Errors

```
RuntimeError: NCCL error
```

**Solutions**:
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # If GPUs are on different PCIe buses
```

## About GGUF Models

The GGUF quantized models from QuantStack/Wan2.2-Animate-14B-GGUF are:
- **Format**: GGUF (GPT-Generated Unified Format)
- **Size**: 18.7 GB (Q8_0 quantization)
- **Compatibility**: NOT directly compatible with official Wan2.2 codebase

The official codebase uses:
- **Format**: Safetensors (Hugging Face format)
- **Multi-GPU**: FSDP (Fully Sharded Data Parallel)
- **Optimization**: Works out-of-the-box with provided scripts

To use GGUF models, you would need to:
1. Convert GGUF → Safetensors format
2. Ensure model architecture matches exactly
3. Test thoroughly (not recommended for production)

**Recommendation**: Use the official safetensors models from Wan-AI/Wan2.2-Animate-14B

## Directory Structure

After setup, your directory structure will look like:

```
/home/caches/
└── Wan2.2-Animate-14B/
    ├── config.json
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    ├── model-00004-of-00004.safetensors
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    ├── relighting_lora.ckpt
    └── process_checkpoint/
        ├── det/yolov10m.onnx
        ├── pose2d/vitpose_h_wholebody.onnx
        └── sam2/sam2_hiera_large.pt

/workspace/Wan2.2/outputs/
└── animate/
    ├── process_results/
    │   ├── src_pose.mp4
    │   ├── src_face.mp4
    │   └── src_ref.png
    └── animate-14B_1280x720_2_*.mp4  (generated output)
```

## Advanced Usage

### Character Replacement Mode

Instead of animating a character, replace one with another:

```bash
# Preprocessing for replacement
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path examples/wan_animate/replace/video.mp4 \
    --refer_path examples/wan_animate/replace/image.jpeg \
    --save_path outputs/animate/replace_results \
    --resolution_area 1280 720 \
    --iterations 3 \
    --k 7 \
    --w_len 1 \
    --h_len 1 \
    --replace_flag

# Run replacement
python -m torch.distributed.run \
    --nproc_per_node 2 \
    generate.py \
    --task animate-14B \
    --ckpt_dir /home/caches/Wan2.2-Animate-14B \
    --src_root_path outputs/animate/replace_results \
    --refert_num 1 \
    --replace_flag \
    --use_relighting_lora \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2
```

### Custom Prompts

While not required, you can customize the prompt:

```bash
python run_wan_animate_multi_gpu.py \
    --video your_video.mp4 \
    --image your_character.jpg
    # The script uses default prompt: "视频中的人在做动作"
```

## Support and Resources

- **Official Repo**: https://github.com/Wan-Video/Wan2.2
- **Paper**: https://arxiv.org/abs/2503.20314
- **HuggingFace Models**: https://huggingface.co/Wan-AI
- **Discord**: https://discord.gg/AKNgpMK4Yj

## License

Wan2.2 models are licensed under Apache 2.0. See LICENSE.txt for details.
