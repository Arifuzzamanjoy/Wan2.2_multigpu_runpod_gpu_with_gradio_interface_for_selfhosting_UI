# 📊 Progress Monitoring Guide

## ✅ What Was Fixed

Your generation was running but you couldn't see progress. Now you have **3 ways** to monitor:

### 1. 🖥️ **Gradio UI Progress Bar** (Real-time)
- Progress bar updates automatically during generation
- Shows current step: "🎬 Generating... Step 5/20"
- No need to do anything - just watch the UI!

### 2. 📺 **Terminal Output** (Real-time)
The terminal where you run `python app_gradio.py` now shows:
- Every sampling step
- GPU memory usage
- NCCL communication logs
- Generation progress in real-time

**Example output you'll see:**
```
🚀 Starting generation command: torchrun --nnodes 1 --nproc_per_node 3...
[INFO] sampling step: 1/20
[INFO] sampling step: 2/20
...
✅ Video generated: /workspace/Wan2.2/outputs/animate/process_results_20251123_081159/output.mp4
```

### 3. 🔍 **Progress Watch Script** (Separate Terminal)
Open a **second terminal** and run:
```bash
cd /workspace/Wan2.2
./watch_generation.sh
```

This will show:
- Which directory is being written to
- Files being created
- When the MP4 file appears
- Auto-refreshes every 5 seconds

---

## 🚀 Quick Start

### Option 1: Run Gradio with Terminal Output
```bash
cd /workspace/Wan2.2
source venv_wan22/bin/activate
python app_gradio.py --share
```

**What you'll see:**
- Gradio starts on port 7860
- When you click "Generate", terminal shows real-time progress
- GPU utilization logs appear
- Step-by-step generation progress

### Option 2: Run + Watch in Split Terminal
**Terminal 1:**
```bash
cd /workspace/Wan2.2
source venv_wan22/bin/activate
python app_gradio.py --share
```

**Terminal 2 (while generating):**
```bash
cd /workspace/Wan2.2
./watch_generation.sh
```

---

## 🔧 Technical Details

### What Changed in `app_gradio.py`:

1. **Subprocess Output Streaming**
   - Changed from `subprocess.run()` to `subprocess.Popen()`
   - Streams stdout/stderr in real-time
   - Prints every line to terminal immediately

2. **Progress Bar Updates**
   - Parses output for "step" keywords
   - Extracts step numbers with regex
   - Updates Gradio progress bar: 0.1 + (0.8 × step/total)

3. **GPU Detection**
   - Auto-detects available GPUs: `torch.cuda.device_count()`
   - Dropdown shows only valid options [1, 2, 3]
   - Default selection matches your system (3 GPUs)

### Sample Progress Flow:
```
0%   → 🚀 Starting generation on 3 GPU(s)...
10%  → ⚙️ Running generation (this may take several minutes)...
15%  → 🎬 Generating... Step 1/20
20%  → 🎬 Generating... Step 2/20
...
90%  → 🔍 Looking for generated video...
100% → ✅ Generation completed!
```

---

## 📋 Monitoring Commands

### Check GPU Usage (Live)
```bash
watch -n 1 nvidia-smi
```

### Check Running Processes
```bash
# See torchrun processes
ps aux | grep torchrun

# See Python generation processes
ps aux | grep generate.py
```

### Check Output Directory
```bash
# List all process results
ls -lth /workspace/Wan2.2/outputs/animate/

# Watch latest directory
watch -n 2 'ls -lh /workspace/Wan2.2/outputs/animate/process_results_* | tail -20'
```

### Monitor Logs
```bash
# If you redirected output to a log file
tail -f generation.log
```

---

## 🎯 Expected Timeline

For a typical video generation with 3 GPUs:

| Step | Time | Description |
|------|------|-------------|
| Preprocessing | 30-60s | Pose detection, face analysis |
| Loading Models | 1-2 min | Load DiT, T5, VAE to GPUs |
| Sampling Steps | 5-10 min | Main generation (20 steps) |
| Saving Video | 10-30s | Encode and save MP4 |
| **Total** | **7-14 min** | Complete pipeline |

---

## ✅ Verification Checklist

After starting generation, you should see:

- [ ] Gradio UI shows progress bar moving
- [ ] Terminal shows `torchrun` command execution
- [ ] Terminal shows sampling step logs
- [ ] `nvidia-smi` shows 3 GPUs at 90%+ utilization
- [ ] Output directory gets created: `process_results_YYYYMMDD_HHMMSS/`
- [ ] MP4 file appears in output directory
- [ ] Video plays in Gradio UI

---

## 🐛 Troubleshooting

### "No progress shown in terminal"
- Make sure you're running `python app_gradio.py` directly (not in background)
- Check that you didn't redirect output: avoid `python app_gradio.py > /dev/null`

### "Progress bar stuck at 10%"
- Generation is still running - check GPU usage with `nvidia-smi`
- Check the watch script in a second terminal
- Sampling steps might not be logging - check output directory for files

### "Process killed / Terminated"
- Out of memory - reduce batch size or use fewer GPUs
- Check `dmesg | tail` for OOM killer messages

### "No output file found"
- Check the exact directory: last line before error shows the path
- List directory: `ls -lh /workspace/Wan2.2/outputs/animate/process_results_*/`

---

## 📝 Notes

- **Terminal output is now verbose** - this is intentional for debugging
- **Progress bar updates** when generation logs contain "step" keywords
- **GPU dropdown auto-adjusts** to your available GPUs (3 in your case)
- **Logs are printed in real-time** - no more silent generation!

---

**Happy Generating! 🎬✨**
