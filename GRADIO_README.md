# 🎬 Wan2.2 Animate - Gradio Web Interface

A beautiful, interactive web interface for generating animated videos using Wan2.2 Animate.

## ✨ Features

- 🎨 **Modern UI** - Clean, intuitive interface with visual feedback
- 🔐 **Authentication** - Optional login protection with username/password
- 📊 **Real-time Progress** - Live status updates during generation
- 🖥️ **Multi-GPU Support** - Automatic configuration for 1-8 GPUs
- 📥 **One-click Setup** - Download models directly from UI
- 🎥 **Instant Preview** - View and download results immediately
- 📱 **Responsive Design** - Works on desktop and tablet

---

## 🚀 Quick Start

### Option 1: Simple Launch (No Authentication)

```bash
./launch_gradio.sh
```

Then open: `http://localhost:7860`

### Option 2: Launch with Authentication (Recommended)

```bash
# Set your credentials (one-time)
export WAN_USERNAME="your_username"
export WAN_PASSWORD="your_secure_password"

# Launch
./launch_with_auth.sh
```

Or use default credentials (admin/wan2024):
```bash
./launch_with_auth.sh
```

### Option 3: Manual Launch with Custom Options

```bash
source ./venv_wan22/bin/activate

# Public access with authentication
python app_gradio.py --share --auth "admin:mypassword"

# Local only, no auth
python app_gradio.py --host 127.0.0.1 --port 7860

# Custom port with auth
python app_gradio.py --port 8080 --auth "user:pass123"
```

---

## 🔐 Authentication Setup

### Method 1: Environment Variables (Recommended)

```bash
# Add to your ~/.bashrc or ~/.profile
export WAN_USERNAME="myuser"
export WAN_PASSWORD="my_secure_pass_2024"

# Then launch
./launch_with_auth.sh
```

### Method 2: Command Line

```bash
python app_gradio.py --auth "username:password" --share
```

### Method 3: Temporary Token

```bash
# Generate a random token
TOKEN=$(openssl rand -hex 16)
echo "Your access token: $TOKEN"

# Launch with token as password
python app_gradio.py --auth "user:$TOKEN" --share
```

**Security Tips:**
- ✅ Use strong passwords (12+ characters)
- ✅ Change default credentials immediately
- ✅ Use `--share` only when needed
- ✅ Set `--host 127.0.0.1` for local-only access
- ❌ Don't use simple passwords like "123456"

---

## ⚙️ Command Line Options

```bash
python app_gradio.py [OPTIONS]

Options:
  --host HOST              Host to bind to (default: 0.0.0.0)
  --port PORT              Port number (default: 7860)
  --share                  Create public share link (for remote access)
  --debug                  Enable debug mode
  --auth USER:PASS         Enable authentication (format: username:password)
  --auth-message MSG       Custom login message
```

### Launch Examples:

**1. Local access only (no authentication):**
```bash
python app_gradio.py --host 127.0.0.1 --port 7860
```

**2. Public access with password:**
```bash
python app_gradio.py --share --auth "admin:SecurePass123!"
```

**3. Custom port with authentication:**
```bash
python app_gradio.py --port 8080 --auth "user:token_xyz"
```

**4. Secure setup with custom message:**
```bash
python app_gradio.py \
    --share \
    --auth "myuser:my_password" \
    --auth-message "Welcome to Wan2.2 Animate - Team Access Only"
```

---

## 🎯 How to Use the Interface

### Step 1: Download Models (First Time Only)

1. Click the **"📥 Download All Models"** button
2. Wait for ~20GB download to complete
3. You'll see "✅ All models downloaded successfully!"

### Step 2: Upload Your Files

**Reference Image Tab:**
- Upload a clear photo of the character/person
- Best: High-resolution, front-facing portraits
- Supported: PNG, JPEG, JPG

**Driving Video Tab:**
- Upload video with the motion you want
- Best: Clear movements, good lighting
- Supported: MP4, MOV, AVI

### Step 3: Configure Settings

**GPU & Performance:**
- **Number of GPUs:** 2 recommended (faster generation)
- **Quality (Steps):** 20 for balanced, 30 for high quality
- **Temporal Consistency:** 5 frames for smoother results

**Advanced Options (Optional):**
- **FLUX:** Better pose transfer (requires FLUX model)
- **Replace Mode:** Character replacement instead of animation
- **Relighting LoRA:** Improved lighting in replace mode

### Step 4: Generate!

1. Click **"🚀 Generate Animated Video"**
2. Watch progress in "Status & Logs" area
3. Wait 5-15 minutes (depending on settings)
4. Preview and download your video!

---

## 🎯 GPU Configuration

The interface supports multi-GPU setups for faster generation:

| GPUs | Ulysses Support | Memory per GPU | Speed | Best For |
|------|----------------|----------------|-------|----------|
| 1    | ✅             | ~40GB          | 1x    | Testing  |
| 2    | ✅             | ~20GB          | ~1.8x | Recommended |
| 4    | ✅             | ~10GB          | ~3x   | Production |
| 5    | ✅             | ~8GB           | ~3.5x | Optimal A40 |
| 8    | ✅             | ~5GB           | ~5x   | High-end |

**Note:** Number of GPUs must divide 40 (num_heads) for Ulysses parallelization.

---

## 📊 Generation Settings

### Sampling Steps
- **10-15:** Fast, lower quality
- **20-30:** Balanced (recommended)
- **30-50:** High quality, slower

### Temporal Guidance Frames (refert_num)
- **1:** Faster generation
- **5:** Better temporal consistency (smoother animations)

### Advanced Options
- **FLUX:** Enhanced pose retargeting (requires FLUX model)
- **Replace Mode:** Character replacement instead of animation
- **Relighting LoRA:** Better lighting in replacement mode

---

## 📁 File Structure

```
/workspace/Wan2.2/
├── app_gradio.py           # Main Gradio interface
├── launch_gradio.sh        # Launch script
├── uploads/                # Uploaded files (auto-created)
├── outputs/
│   └── animate/
│       ├── process_results_*/  # Preprocessed data
│       └── *.mp4              # Generated videos
└── /home/caches/
    └── Wan2.2-Animate-14B/    # Downloaded models
```

---

## 🔧 Troubleshooting

### Models not downloading?
```bash
# Manual download
python -c "from app_gradio import download_models; download_models()"
```

### Port already in use?
```bash
# Use a different port
python app_gradio.py --port 8080
```

### GPU out of memory?
- Reduce number of GPUs (use 1 or 2)
- Lower sampling steps (use 15-20)
- Use refert_num=1 instead of 5

### Connection refused?
```bash
# Check if app is running
ps aux | grep app_gradio

# Check port
netstat -tuln | grep 7860
```

---

## 🎨 Example Workflow

1. **Start the server:**
   ```bash
   ./launch_gradio.sh
   ```

2. **Open browser:** `http://localhost:7860`

3. **First-time setup:**
   - Click "Download Models" button
   - Wait for download to complete (~20GB)

4. **Generate video:**
   - Upload reference image (e.g., portrait photo)
   - Upload driving video (e.g., dance video)
   - Select 2 GPUs, 20 steps, refert_num=1
   - Click "Generate Video"
   - Wait ~5-10 minutes
   - Download result!

---

## 📝 Performance Tips

### For Faster Generation:
- Use 2+ GPUs
- Set sampling steps to 15-20
- Use refert_num=1
- Keep videos short (<10 seconds)

### For Better Quality:
- Use 4-8 GPUs
- Set sampling steps to 25-30
- Use refert_num=5
- Enable FLUX preprocessing

### Memory Optimization:
- **48GB GPUs:** Use 2 GPUs (optimal)
- **80GB GPUs:** Use 4-8 GPUs
- **Limited RAM:** Use refert_num=1

---

## 🐛 Common Issues

### Issue: "Models not downloaded"
**Solution:** Click the "Download Models" button first

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce number of GPUs to 1 or 2
- Lower sampling steps to 15
- Use refert_num=1

### Issue: "Video not found after generation"
**Solution:**
- Check `/workspace/Wan2.2/outputs/animate/process_results_*/`
- Look for .mp4 files
- Check generation logs for errors

### Issue: "Preprocessing failed"
**Solution:**
- Ensure preprocessing models are downloaded
- Check video format (MP4 recommended)
- Check image format (PNG/JPEG)

---

## 📞 Support

For issues or questions:
1. Check the logs in the terminal
2. Review the troubleshooting section above
3. Check GPU memory usage: `nvidia-smi`
4. Verify models are downloaded: `ls /home/caches/Wan2.2-Animate-14B/`

---

## 🎉 Credits

- **Wan2.2 Animate:** Wan-AI/Wan2.2-Animate-14B
- **Gradio Interface:** Custom UI for easy video generation
- **Multi-GPU Support:** FSDP + Ulysses parallelization

---

**Enjoy creating amazing animated videos! 🎬✨**
