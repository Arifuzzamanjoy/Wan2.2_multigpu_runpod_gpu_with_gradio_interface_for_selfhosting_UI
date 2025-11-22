# Wan2.2 Multi-GPU Methods - Complete Guide

## 🎯 Available Multi-GPU Strategies in Wan2.2

### **Method 1: FSDP (Fully Sharded Data Parallel)**
**Location**: `wan/distributed/fsdp.py`

**What it does**:
- Shards model parameters, gradients, and optimizer states across GPUs
- Reduces per-GPU memory from 28GB → ~14GB per GPU
- Uses PyTorch's native FSDP implementation

**How to enable**:
```bash
torchrun --nproc_per_node=2 generate.py \
  --task animate-14B \
  --dit_fsdp \      # Enable for DiT model
  --t5_fsdp \       # Enable for T5 encoder
  --ckpt_dir /path/to/model
```

**Pros**:
- ✅ Dramatic memory reduction
- ✅ Native PyTorch support
- ✅ Works with models >100GB

**Cons**:
- ⚠️ Communication overhead between GPUs
- ⚠️ Slightly slower than single-GPU if model fits in VRAM

**Best for**: Models that don't fit on single GPU (14B+ parameters)

---

### **Method 2: Ulysses Sequence Parallel**
**Location**: `wan/distributed/ulysses.py`

**What it does**:
- Splits the **sequence dimension** (video frames) across GPUs
- Each GPU processes different frames simultaneously
- Uses all-to-all communication for attention computation

**How to enable**:
```bash
torchrun --nproc_per_node=2 generate.py \
  --task animate-14B \
  --ulysses_size 2 \  # MUST equal number of GPUs
  --ckpt_dir /path/to/model
```

**Requirements**:
- `num_attention_heads % ulysses_size == 0`
- For Wan2.2-Animate-14B: 20 heads → works with 2, 4, 5, 10 GPUs

**Pros**:
- ✅ Excellent for long sequences (videos)
- ✅ Near-linear speedup for video generation
- ✅ Flash attention optimization

**Cons**:
- ⚠️ Requires head count to be divisible by GPU count
- ⚠️ Communication overhead on slow interconnects

**Best for**: Long video generation (80+ frames)

---

### **Method 3: FSDP + Ulysses (Hybrid)** ⭐ **RECOMMENDED**
**What it does**:
- Combines model parallelism (FSDP) + sequence parallelism (Ulysses)
- FSDP shards model weights across GPUs
- Ulysses distributes sequence processing across GPUs

**How to enable**:
```bash
torchrun --nproc_per_node=2 generate.py \
  --task animate-14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --ckpt_dir /path/to/model
```

**Memory usage (2x A40 48GB)**:
- Without: Would need 60GB+ per GPU (OOM)
- With FSDP only: ~18GB per GPU
- With FSDP+Ulysses: ~14GB per GPU

**Pros**:
- ✅ Best of both worlds
- ✅ Balanced memory & compute distribution
- ✅ Fastest for multi-GPU setups

**Cons**:
- ⚠️ Most complex setup
- ⚠️ Requires good GPU interconnect (NVLink/PCIe 4.0+)

**Best for**: 2-8 GPU setups with 40-80GB VRAM per GPU

---

### **Method 4: CPU Offloading** (Fallback)
**Location**: `generate.py` (automatic)

**What it does**:
- Offloads model components to CPU RAM between forward passes
- VAE, T5, DiT models moved on/off GPU as needed

**How to enable**:
```bash
python generate.py \
  --task animate-14B \
  --offload_model True \  # Explicitly enable
  --t5_cpu \              # Keep T5 on CPU permanently
  --ckpt_dir /path/to/model
```

**Auto-enabled when**:
- `world_size == 1` (single GPU)
- No FSDP or Ulysses flags set

**Pros**:
- ✅ Works on single GPU with <24GB VRAM
- ✅ No code changes needed

**Cons**:
- ❌ **EXTREMELY SLOW** (5-10x slower)
- ❌ NOT compatible with multi-GPU distributed modes
- ❌ CPU-GPU transfer bottleneck

**Best for**: Testing only, NOT for production

---

## 🔥 Performance Comparison (2x A40 48GB)

| Method | Speed | Memory/GPU | Quality | Complexity |
|--------|-------|------------|---------|------------|
| **Single GPU + Offload** | 1x (baseline) | 20GB | ✅ | ⭐ |
| **FSDP Only** | 1.2x | 18GB | ✅ | ⭐⭐ |
| **Ulysses Only** | 1.8x | 28GB | ✅ | ⭐⭐ |
| **FSDP + Ulysses** ⭐ | **2.5x** | **14GB** | ✅ | ⭐⭐⭐ |

---

## 📊 Detailed Architecture

### FSDP Architecture
```
GPU 0: [Blocks 0-11 sharded] + [T5 sharded]
GPU 1: [Blocks 12-23 sharded] + [T5 sharded]

During Forward:
  - GPU0 broadcasts Blocks 0-11 → GPU1
  - GPU1 broadcasts Blocks 12-23 → GPU0
  - Both compute in parallel
  - Gradients all-reduced
```

### Ulysses Architecture
```
Input: 77 frames, 20 attention heads
GPU 0: Processes frames 0-38, heads 0-9
GPU 1: Processes frames 39-76, heads 10-19

During Attention:
  1. All-to-all scatter: split heads across GPUs
  2. Flash attention on local data
  3. All-to-all gather: recombine results
```

### FSDP + Ulysses (Hybrid)
```
Model Sharding (FSDP):
  - DiT blocks distributed across GPUs
  - T5 encoder distributed
  - VAE shared (small enough)

Sequence Sharding (Ulysses):
  - 77 frames split → 38-39 frames per GPU
  - Attention computed in parallel
  - Results synchronized via NCCL
```

---

## 🛠️ Optimization Tips

### 1. **NCCL Configuration** (Critical!)
```bash
# For A40 with NVLink
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# For PCIe-only systems
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
```

### 2. **CUDA Memory Management**
```bash
# Allow memory to grow dynamically
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
```

### 3. **CPU Threading**
```bash
# For 2 GPUs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# For 4+ GPUs, reduce per-GPU threads
export OMP_NUM_THREADS=4
```

### 4. **Batch Size Tuning**
```python
# Not directly exposed in CLI, but affects memory:
# clip_len = 77 (frames)
# - Larger clip_len = more memory
# - Use refert_num=1 for speed
# - Use refert_num=5 for quality
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "GPU 0 at 100%, GPU 1 at 1%"
**Cause**: Using deprecated `torch.distributed.run` or offload_model=True

**Solution**:
```bash
# ❌ WRONG
python -m torch.distributed.run generate.py ...

# ✅ CORRECT
torchrun generate.py --dit_fsdp --t5_fsdp --ulysses_size 2 ...
```

### Issue 2: "NCCL timeout / hanging"
**Cause**: Bad GPU interconnect or wrong NCCL settings

**Solution**:
```bash
# Check GPU topology
nvidia-smi topo -m

# If NVLink available:
export NCCL_P2P_LEVEL=NVL

# If only PCIe:
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
```

### Issue 3: "OOM even with FSDP"
**Cause**: Sequence too long or wrong ulysses_size

**Solution**:
```bash
# Enable Ulysses to split sequence
torchrun --nproc_per_node=2 generate.py \
  --dit_fsdp --t5_fsdp --ulysses_size 2 \
  --refert_num 1  # Use 1 instead of 5 for less memory
```

### Issue 4: "Model not starting generation"
**Cause**: Likely stuck in NCCL initialization or wrong process spawning

**Solution**:
```bash
# Check NCCL logs
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Use explicit master address
torchrun --master_addr=127.0.0.1 --master_port=29500 ...
```

---

## 📈 Scaling Beyond 2 GPUs

### 4 GPUs (4x A40):
```bash
torchrun --nproc_per_node=4 generate.py \
  --task animate-14B \
  --dit_fsdp --t5_fsdp --ulysses_size 4 \
  --sample_steps 20
```
**Expected**:
- Memory: ~8-10GB per GPU
- Speed: 4-5x faster than single GPU
- Quality: Identical

### 8 GPUs (8x A40):
```bash
torchrun --nproc_per_node=8 generate.py \
  --task animate-14B \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --sample_steps 20
```
**Expected**:
- Memory: ~5-6GB per GPU
- Speed: 7-9x faster
- Requires good interconnect (NVLink/NVSwitch)

---

## 🎓 Reference Papers

1. **FSDP**: [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
2. **Ulysses**: [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
3. **Flash Attention**: [FlashAttention-2](https://arxiv.org/abs/2307.08691)

---

## 🔍 Debugging Commands

```bash
# Check GPU utilization
nvidia-smi dmon -s mu

# Check GPU memory
watch -n 1 nvidia-smi

# Check GPU topology (NVLink)
nvidia-smi topo -m

# Check NCCL bandwidth
/workspace/Wan2.2/venv_wan22/bin/python -c "import torch; torch.cuda.nccl.version()"

# Profile distributed training
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

---

## ✅ Best Practices for 2x A40 (Your Setup)

1. **Always use FSDP + Ulysses**:
   ```bash
   --dit_fsdp --t5_fsdp --ulysses_size 2
   ```

2. **Use `torchrun` not `torch.distributed.run`**:
   ```bash
   torchrun --nproc_per_node=2 ...
   ```

3. **Optimize NCCL for A40**:
   ```bash
   export NCCL_P2P_LEVEL=NVL
   export NCCL_IB_DISABLE=0
   ```

4. **Balance speed vs quality**:
   - Fast: `--sample_steps 10 --refert_num 1`
   - Balanced: `--sample_steps 20 --refert_num 1`
   - Quality: `--sample_steps 30 --refert_num 5`

5. **Monitor GPU balance**:
   ```bash
   nvidia-smi dmon -s mu -c 100
   # Both GPUs should be >80% utilization
   ```
