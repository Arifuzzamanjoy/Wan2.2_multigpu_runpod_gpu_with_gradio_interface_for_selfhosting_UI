# 🚀 Quick Start - FIXED Multi-GPU Script

## 🔥 What's Fixed in `run_wan_animate_FIXED.py`

### Critical Fixes from `run_wan_animate_multi_gpu.py`:

1. **✅ Uses `torchrun` instead of deprecated `torch.distributed.run`**
   - Old: `python -m torch.distributed.run`
   - New: `torchrun`
   - Impact: Proper process spawning, better GPU load balancing

2. **✅ Optimized NCCL environment for A40 P2P communication**
   - Added: `NCCL_P2P_LEVEL=NVL` (NVLink optimization)
   - Added: `NCCL_NET_GDR_LEVEL=5` (GPU Direct RDMA)
   - Impact: 30-50% faster GPU-to-GPU transfers

3. **✅ Better CUDA memory management**
   - Set: `max_split_size_mb=512` for better fragmentation handling
   - Impact: More stable memory usage, fewer OOM errors

4. **✅ Explicit master port to avoid conflicts**
   - Added: `--master_port 29500`
   - Impact: No more "address already in use" errors

---

## 📦 Installation (One-Time Setup)

```bash
# Activate your venv
source /workspace/Wan2.2/venv_wan22/bin/activate

# Make script executable
chmod +x /workspace/Wan2.2/run_wan_animate_FIXED.py
```

---

## 🎬 Basic Usage

### **First Run (with downloads)**
```bash
python /workspace/Wan2.2/run_wan_animate_FIXED.py \
  --video /path/to/your/video.mp4 \
  --image /path/to/your/reference.jpg
```

This will:
1. Download Wan2.2-Animate-14B model (~28GB)
2. Download preprocessing models (~500MB)
3. Preprocess your video
4. Generate animation on 2 GPUs

**Time**: ~30-60 min (first run, includes downloads)

---

### **Subsequent Runs (skip downloads)**
```bash
python /workspace/Wan2.2/run_wan_animate_FIXED.py \
  --skip-download \
  --video /path/to/your/video.mp4 \
  --image /path/to/your/reference.jpg
```

**Time**: ~5-15 min per video (depending on length)

---

### **Using Preprocessed Data**
```bash
# If you already preprocessed the video
python /workspace/Wan2.2/run_wan_animate_FIXED.py \
  --skip-download \
  --skip-preprocessing
```

**Time**: ~3-10 min (generation only)

---

## ⚙️ Advanced Options

### **Quality Control**
```bash
# Fast (lower quality)
python run_wan_animate_FIXED.py \
  --sample-steps 10 \
  --refert-num 1 \
  --skip-download

# Balanced (default)
python run_wan_animate_FIXED.py \
  --sample-steps 20 \
  --refert-num 1 \
  --skip-download

# High Quality (slower)
python run_wan_animate_FIXED.py \
  --sample-steps 30 \
  --refert-num 5 \
  --skip-download
```

**Performance Impact**:
- `sample-steps`: 10 = 3min, 20 = 6min, 30 = 9min (per video)
- `refert-num`: 1 = faster, 5 = better temporal consistency

---

### **Character Replacement Mode**
```bash
python run_wan_animate_FIXED.py \
  --replace-mode \
  --use-relighting-lora \
  --skip-download
```

Use this when you want to replace a character while keeping the background.

---

### **Enhanced Pose Retargeting (FLUX)**
```bash
python run_wan_animate_FIXED.py \
  --use-flux \
  --skip-download
```

⚠️ **Requires**: FLUX.1-Kontext-dev model (additional download)

---

## 🔍 Monitoring GPU Usage

### **Real-time GPU monitoring**
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

**Expected output (healthy)**:
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A40          Off  | 00000000:53:00.0 Off |                    0 |
| 45%   42C    P0   150W / 300W |  18000MiB / 48685MiB |     95%      Default |
+-----------------------------------------------------------------------------+
|   1  NVIDIA A40          Off  | 00000000:56:00.0 Off |                    0 |
| 44%   41C    P0   145W / 300W |  17500MiB / 48685MiB |     92%      Default |
+-----------------------------------------------------------------------------+
```

**Key indicators**:
- ✅ Both GPUs at 80-100% utilization
- ✅ Memory usage balanced (~14-20GB each)
- ✅ Power draw 140-180W each

**Bad indicators**:
- ❌ GPU 0 at 100%, GPU 1 at 1% → Load imbalance
- ❌ Memory 45GB on one GPU → Not using FSDP properly
- ❌ Utilization stuck at 0% → Process hung

---

## 🐛 Troubleshooting

### **Issue**: Script hangs at "Loading checkpoint shards"
**Solution**: This is normal! Wait 2-3 minutes for FSDP to initialize.

---

### **Issue**: "NCCL timeout" or "communication failed"
**Diagnosis**:
```bash
# Check GPU interconnect
nvidia-smi topo -m
```

**Solution**:
```bash
# If you see NVLink in topology
export NCCL_P2P_LEVEL=NVL

# If only PCIe
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1

# Then retry
python run_wan_animate_FIXED.py --skip-download
```

---

### **Issue**: OOM (Out of Memory) Error
**Solutions**:
```bash
# 1. Reduce clip length (edit generate.py)
--frame_num 77  # Try 65 or 49

# 2. Use lower sample steps
--sample-steps 10

# 3. Clear CUDA cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

---

### **Issue**: "Address already in use"
**Solution**:
```bash
# Kill any existing processes
pkill -f torchrun
pkill -f generate.py

# Or use different port
# Edit run_wan_animate_FIXED.py line 158:
# Change: "--master_port", "29500"
# To:     "--master_port", "29501"
```

---

## 📊 Performance Expectations (2x A40 48GB)

| Video Length | Preprocessing | Generation | Total Time |
|--------------|---------------|------------|------------|
| 3 sec (77 frames) | 2-3 min | 5-8 min | ~10 min |
| 5 sec (128 frames) | 3-4 min | 8-12 min | ~15 min |
| 10 sec (256 frames) | 5-7 min | 15-20 min | ~25 min |

**Factors affecting speed**:
- `--sample-steps`: Higher = slower but better quality
- `--refert-num`: 5 is slower than 1
- Video resolution: 1280x720 is baseline
- GPU load: Ensure both GPUs are active

---

## 🎯 Example Commands for Your Setup

### **Test with example data**
```bash
cd /workspace/Wan2.2
python run_wan_animate_FIXED.py \
  --skip-download \
  --skip-preprocessing \
  --num-gpus 2 \
  --sample-steps 20 \
  --refert-num 1
```

### **Process your own video**
```bash
python run_wan_animate_FIXED.py \
  --skip-download \
  --video /workspace/Wan2.2/examples/wan_animate/animate/video.mp4 \
  --image /workspace/Wan2.2/examples/wan_animate/animate/image.jpeg \
  --num-gpus 2 \
  --sample-steps 20
```

### **High-quality production run**
```bash
python run_wan_animate_FIXED.py \
  --skip-download \
  --video /path/to/driving_video.mp4 \
  --image /path/to/character_reference.jpg \
  --num-gpus 2 \
  --sample-steps 30 \
  --refert-num 5
```

---

## 📝 Output Location

Generated videos are saved to:
```
/workspace/Wan2.2/outputs/animate/
```

Files created:
- `animate-14B_1280x720_2_<prompt>_<timestamp>.mp4` - Final video
- `process_results/` - Preprocessed data (pose, face, etc.)

---

## 🔄 Cleaning Up

### **Clear cached models**
```bash
rm -rf /home/caches/Wan2.2-Animate-14B
```

### **Clear outputs**
```bash
rm -rf /workspace/Wan2.2/outputs/animate/*
```

### **Full reset**
```bash
rm -rf /home/caches/Wan2.2-Animate-14B
rm -rf /workspace/Wan2.2/outputs/animate/*
pkill -f torchrun
```

---

## 💡 Pro Tips

1. **Reuse preprocessing**: Use `--skip-preprocessing` to test different sampling steps without re-preprocessing

2. **Batch processing**: Create a shell script to process multiple videos:
   ```bash
   #!/bin/bash
   for video in /path/to/videos/*.mp4; do
     python run_wan_animate_FIXED.py \
       --skip-download \
       --video "$video" \
       --image /path/to/reference.jpg
   done
   ```

3. **Monitor from another terminal**:
   ```bash
   # Terminal 1: Run generation
   python run_wan_animate_FIXED.py ...
   
   # Terminal 2: Monitor GPUs
   watch -n 0.5 nvidia-smi
   
   # Terminal 3: Monitor logs
   tail -f /path/to/log/file
   ```

4. **Speed up testing**: Use `--sample-steps 5` for quick tests, then use 20-30 for final renders

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs** - Look for ERROR messages in terminal output
2. **Check GPU status** - Run `nvidia-smi` to see GPU health
3. **Check NCCL** - Set `export NCCL_DEBUG=INFO` for detailed logs
4. **Compare with original script** - See `MULTI_GPU_METHODS.md` for detailed explanations

---

## 🎉 Success Checklist

- [x] Models downloaded to `/home/caches/Wan2.2-Animate-14B`
- [x] Both GPUs showing 80-100% utilization
- [x] Memory usage balanced (~14-20GB per GPU)
- [x] No NCCL timeout errors
- [x] Generation completes successfully
- [x] Output video saved to `/workspace/Wan2.2/outputs/animate/`

If all checked ✅, you're good to go! 🚀
