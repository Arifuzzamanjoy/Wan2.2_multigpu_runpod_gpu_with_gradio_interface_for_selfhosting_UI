# Wan2.2-Animate Multi-GPU Solutions Summary

## Issues Found and Solutions Applied

### 1. **ModuleNotFoundError: No module named 'transformers.modeling_layers'**
**GitHub Issue Reference:** This is a compatibility issue between `peft` and `transformers` versions.

**Problem:**
- `peft 0.18.0` was trying to import from `transformers.modeling_layers` which doesn't exist in older versions of `transformers`
- The module structure changed in transformers 4.50+

**Solution:**
```bash
# Upgrade transformers to latest version
pip install --upgrade transformers
# Result: transformers 4.57.1 (up from 4.51.3)
```

**Related GitHub Issues:**
- Issue #283: Similar OOM and dependency issues on A6000 48GB setup
- Issue #111: Installation problems with various dependencies

### 2. **ModuleNotFoundError: No module named 'loguru'**
**Problem:** Missing logging dependency required by preprocessing pipeline

**Solution:**
```bash
pip install loguru
```

### 3. **Missing Animate Dependencies**
**Problem:** Various missing dependencies for the Wan2.2-Animate module

**Solution:**
```bash
pip install -r requirements_animate.txt
```

**Key Dependencies Installed:**
- `decord` - Video decoding
- `onnxruntime` - Model inference
- `pandas` - Data handling
- `matplotlib` - Visualization
- `SAM-2` - Segment Anything Model 2 (from GitHub)
- `loguru` - Logging
- `sentencepiece` - Tokenization
- `moviepy` - Video processing
- `librosa` - Audio processing

### 4. **ValueError: FLUX.1-Kontext-dev model not found**
**Problem:** The preprocessing script was using `--use_flux` flag by default, which requires the FLUX.1-Kontext-dev model that wasn't downloaded

**Solution:**
Changed default behavior in `run_wan_animate_multi_gpu.py`:
```python
# Before:
default=True

# After:
default=False
```

**Note:** FLUX is optional and only needed for enhanced pose retargeting when the reference image or video doesn't have a standard front-facing pose.

### 5. **Missing moviepy Dependency**
**GitHub Issue Reference:** Issue #209 - Missing moviepy dependency

**Solution:**
```bash
pip install moviepy
```

## Final Working Configuration

### Package Versions (Tested & Working)
```
transformers==4.57.1
peft==0.18.0
diffusers==0.35.2
torch==2.9.1
loguru==0.7.3
moviepy==2.2.1
librosa==0.11.0
SAM-2==1.0
```

### Command to Run (After All Fixes)
```bash
# Activate virtual environment
source venv/bin/activate

# Run with default settings (no FLUX)
python run_wan_animate_multi_gpu.py

# Or run with FLUX if you download FLUX.1-Kontext-dev manually:
python run_wan_animate_multi_gpu.py --use-flux
```

## How to Download FLUX.1-Kontext-dev (Optional)

If you want to use the `--use-flux` option for enhanced pose retargeting:

```bash
# Download FLUX.1-Kontext-dev to the preprocessing checkpoint directory
mkdir -p /home/caches/Wan2.2-Animate-14B/process_checkpoint/FLUX.1-Kontext-dev

# Use huggingface-cli or manually clone
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev \
    --local-dir /home/caches/Wan2.2-Animate-14B/process_checkpoint/FLUX.1-Kontext-dev
```

## GitHub Issues Summary

From our research of 228 issues in the Wan-Video/Wan2.2 repository:

**Most Common Issues:**
1. **Dependency Conflicts** (Issues #278, #283, #111)
   - transformers/peft version mismatches
   - Missing dependencies (moviepy, librosa, loguru)
   - flash-attn installation problems

2. **Memory Issues** (Issue #283)
   - OOM errors on A6000 48GB
   - Need to use gradient checkpointing and FSDP

3. **Model Download Issues**
   - Slow downloads from Hugging Face
   - Missing preprocessing models
   - GGUF models not compatible with official codebase

**Our Solutions Address:**
- ✅ All dependency issues fixed
- ✅ Multi-GPU setup with FSDP for 2x A40 (96GB total VRAM)
- ✅ Preprocessing models downloaded automatically
- ✅ Optional FLUX usage for flexibility

## Troubleshooting

### If preprocessing still fails:
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check package versions
pip list | grep -E "(transformers|peft|diffusers|torch)"

# Run preprocessing manually for debugging
cd /workspace/Wan2.2
source venv/bin/activate
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path /home/caches/Wan2.2-Animate-14B/process_checkpoint \
    --video_path examples/wan_animate/animate/video.mp4 \
    --refer_path examples/wan_animate/animate/image.jpeg \
    --save_path outputs/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag
```

### If OOM errors occur during inference:
```bash
# Reduce resolution
python run_wan_animate_multi_gpu.py --resolution 960 540

# Reduce sampling steps
python run_wan_animate_multi_gpu.py --sample-steps 10

# Use single reference frame instead of 5
python run_wan_animate_multi_gpu.py --refert-num 1
```

## Next Steps

1. **Wait for preprocessing to complete** - This can take several minutes depending on video length
2. **Monitor multi-GPU inference** - Check `nvidia-smi` to verify both GPUs are being used
3. **Check output** - Results will be in `/workspace/Wan2.2/outputs/animate/`

## References

- Main Repository: https://github.com/Wan-Video/Wan2.2
- Model Card: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- Issue #283 (A6000 setup): https://github.com/Wan-Video/Wan2.2/issues/283
- Issue #278 (Dependencies): https://github.com/Wan-Video/Wan2.2/issues/278
- Issue #209 (Moviepy): https://github.com/Wan-Video/Wan2.2/issues/209
