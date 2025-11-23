# 🚀 Wan2.2 Animate - Command Reference

**Complete list of all successful commands for quick reference**

---

## 📦 Installation & Setup

### 1. Install Gradio
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
pip install gradio
```

### 2. Git Configuration
```bash
git config --global user.name "Arifuzzamanjoy"
git config --global user.email "s1710374103@ru.ac.bd"
```

---

## 🎬 Running Gradio Interface

### Option 1: Quick Start (Simplest)
```bash
cd /workspace/Wan2.2
./start.sh
```

### Option 2: Simple Launch
```bash
cd /workspace/Wan2.2
./launch_gradio.sh
```

### Option 3: With Authentication (Recommended)
```bash
cd /workspace/Wan2.2
./launch_with_auth.sh
# Default credentials: admin/wan2024
```

### Option 4: Interactive Menu
```bash
cd /workspace/Wan2.2
./demo_launch.sh
```

### Option 5: Manual Launch
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python app_gradio.py --host 0.0.0.0 --port 7860
```

### Option 6: With Public Share Link
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python app_gradio.py --host 0.0.0.0 --port 7860 --share
```

### Option 7: With Custom Authentication
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python app_gradio.py --auth "username:password" --share
```

### Option 8: With Environment Variables Auth
```bash
export WAN_USERNAME="myuser"
export WAN_PASSWORD="mypassword"
cd /workspace/Wan2.2
./launch_with_auth.sh
```

---

## 🛑 Stop Gradio Server

```bash
pkill -f "python app_gradio.py"
```

---

## 🔍 Check Status

### Check if Gradio is Running
```bash
ps aux | grep app_gradio | grep -v grep
```

### Check Port Usage
```bash
netstat -tuln | grep 7860
```

### Test HTTP Connection
```bash
curl http://localhost:7860
```

---

## 🧪 Testing & Validation

### Check Python Syntax
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python -m py_compile app_gradio.py
```

### Test Model Detection
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python -c "
from app_gradio import check_models_downloaded
result = check_models_downloaded()
print(f'Models downloaded: {result}')
"
```

### Check Gradio Version
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python -c "import gradio; print('Gradio version:', gradio.__version__)"
```

### Check CUDA/PyTorch
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Check GPU Memory
```bash
nvidia-smi
```

---

## 📂 File Management

### List Model Files
```bash
ls -lh /home/caches/Wan2.2-Animate-14B/
```

### Check Disk Space
```bash
df -h
```

### Clear Upload Cache
```bash
rm -rf /workspace/Wan2.2/uploads/*
```

### Clear Output Results
```bash
rm -rf /workspace/Wan2.2/outputs/animate/process_results_*
```

---

## 🔐 Security & Authentication

### Generate Random Password
```bash
openssl rand -hex 16
```

### Generate Random Token
```bash
openssl rand -base64 32
```

### Launch with Generated Token
```bash
TOKEN=$(openssl rand -hex 16)
echo "Access token: $TOKEN"
python app_gradio.py --auth "user:$TOKEN" --share
```

---

## 🐙 Git Commands

### Check Git Status
```bash
cd /workspace/Wan2.2
git status
```

### Stage All Changes
```bash
cd /workspace/Wan2.2
git add -A
```

### Commit Changes
```bash
cd /workspace/Wan2.2
git commit -m "Your commit message"
```

### Push to GitHub
```bash
cd /workspace/Wan2.2
git push myrepo main
```

### View Recent Commits
```bash
cd /workspace/Wan2.2
git log --oneline -5
```

### Check Remote URLs
```bash
cd /workspace/Wan2.2
git remote -v
```

---

## 🎥 Generate Videos (Command Line)

### Run Multi-GPU Generation
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python run_wan_animate_FIXED.py \
    --video /path/to/video.mp4 \
    --image /path/to/image.jpg \
    --num-gpus 2 \
    --sample-steps 20 \
    --refert-num 1
```

### Skip Model Download (if already downloaded)
```bash
python run_wan_animate_FIXED.py \
    --skip-download \
    --video /path/to/video.mp4 \
    --image /path/to/image.jpg \
    --num-gpus 2
```

### Skip Preprocessing (if already preprocessed)
```bash
python run_wan_animate_FIXED.py \
    --skip-download \
    --skip-preprocessing \
    --num-gpus 2
```

---

## 🛠️ Troubleshooting

### Restart Gradio (if frozen)
```bash
pkill -f "python app_gradio.py"
cd /workspace/Wan2.2
./start.sh
```

### Clear Python Cache
```bash
find /workspace/Wan2.2 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

### Check Python Environment
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
which python
python --version
```

### Test Import
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
python -c "import gradio, torch, transformers; print('All imports OK')"
```

### Check NCCL (for multi-GPU)
```bash
python -c "import torch; print(f'NCCL available: {torch.cuda.nccl.is_available()}')"
```

---

## 📊 Monitoring

### Monitor GPU Usage (live)
```bash
watch -n 1 nvidia-smi
```

### Monitor Gradio Logs (if running in background)
```bash
tail -f /var/log/gradio.log  # if logging enabled
```

### Check System Resources
```bash
htop  # or top
```

---

## 🚀 Production Deployment

### Launch as Background Service
```bash
cd /workspace/Wan2.2
source ./venv_wan22/bin/activate
nohup python app_gradio.py --host 0.0.0.0 --port 7860 --auth "admin:password" > gradio.log 2>&1 &
```

### Check Background Process
```bash
ps aux | grep app_gradio
```

### View Logs
```bash
tail -f gradio.log
```

### Stop Background Service
```bash
pkill -f "python app_gradio.py"
```

---

## 📝 Quick URLs

### Access Points
- **Local:** http://localhost:7860
- **Network:** http://0.0.0.0:7860
- **Public (with --share):** https://xxxxx.gradio.live

---

## 🎯 Complete Workflow Example

```bash
# 1. Navigate to workspace
cd /workspace/Wan2.2

# 2. Activate environment
source ./venv_wan22/bin/activate

# 3. Set authentication (optional)
export WAN_USERNAME="admin"
export WAN_PASSWORD="secure_password_2024"

# 4. Launch Gradio
./launch_with_auth.sh

# 5. Open browser to: http://localhost:7860

# 6. Login and use the interface

# 7. When done, stop server
# Press Ctrl+C or in new terminal:
pkill -f "python app_gradio.py"
```

---

## 💡 Useful Aliases (Add to ~/.bashrc)

```bash
# Add these to your ~/.bashrc for quick access
alias wan-start='cd /workspace/Wan2.2 && ./start.sh'
alias wan-stop='pkill -f "python app_gradio.py"'
alias wan-status='ps aux | grep app_gradio | grep -v grep'
alias wan-gpu='nvidia-smi'
alias wan-logs='tail -f /workspace/Wan2.2/gradio.log'
```

Then reload: `source ~/.bashrc`

---

## 📞 Emergency Commands

### Force Kill All Python Processes (USE WITH CAUTION!)
```bash
pkill -9 python
```

### Clear All CUDA Memory
```bash
nvidia-smi --gpu-reset
```

### Restart CUDA Service (if needed)
```bash
sudo systemctl restart nvidia-persistenced
```

---

**Last Updated:** November 23, 2025  
**Repository:** https://github.com/Arifuzzamanjoy/Wan2.2_multigpu_runpod_gpu_with_gradio_interface_for_selfhosting_UI.git

**Questions?** Check GRADIO_README.md and USAGE_GUIDE.md for more details.
