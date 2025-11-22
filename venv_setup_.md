# Virtual Environment Setup Guide for Wan2.2

This guide helps you recreate the `venv_wan22` virtual environment in a different location.

## Environment Overview

- **Python Version**: 3.10
- **CUDA Version**: 12.1
- **Primary Framework**: PyTorch 2.4.0

## Quick Setup

### 1. Create Virtual Environment

```bash
python3.10 -m venv venv_wan22
source venv_wan22/bin/activate  # Linux/Mac
# or
venv_wan22\Scripts\activate  # Windows
```

### 2. Install Core Dependencies

#### Option A: Using Project Requirements Files (Recommended)
```bash
# Install main requirements
pip install -r requirements.txt

# For animate functionality
pip install -r requirements_animate.txt

# For speech-to-video
pip install -r requirements_s2v.txt
```

#### Option B: Manual Installation of Core Packages

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install core ML/AI packages
pip install transformers==4.51.3 diffusers==0.35.2 accelerate==1.11.0
pip install flash-attn==2.8.3 peft==0.17.1
pip install safetensors==0.6.2 huggingface-hub==0.35.3


########for matcthing



pip install tokenizers>=0.20.3 accelerate>=1.1.1 easydict ftfy 
pip install tqdm imageio[ffmpeg] dashscope imageio-ffmpeg numpy>=1.23.5,<2 





# Install ModelScope ecosystem
pip install modelscope==1.31.0 dashscope==1.24.7

# Install video/audio processing
pip install decord==0.6.0 opencv-python==4.11.0.86 imageio==2.37.0 imageio-ffmpeg==0.6.0
pip install librosa==0.11.0 soundfile==0.13.1 openai-whisper==20250625

# Install utilities
pip install einops==0.8.1 omegaconf==2.3.0 hydra-core==1.3.2
pip install pytorch-lightning==2.5.5 torchmetrics==1.8.2
```

## Complete Package List

### Core ML/AI Frameworks
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.4.0+cu121 | PyTorch deep learning framework |
| torchvision | 0.19.0+cu121 | Computer vision utilities |
| torchaudio | 2.4.0+cu121 | Audio processing |
| transformers | 4.51.3 | Hugging Face transformers |
| diffusers | 0.35.2 | Diffusion models |
| flash-attn | 2.8.3 | Flash Attention optimization |

### Training & Optimization
| Package | Version | Purpose |
|---------|---------|---------|
| accelerate | 1.11.0 | Distributed training |
| pytorch-lightning | 2.5.5 | Training framework |
| peft | 0.17.1 | Parameter-efficient fine-tuning |
| torchmetrics | 1.8.2 | Metrics computation |

### Video/Audio Processing
| Package | Version | Purpose |
|---------|---------|---------|
| decord | 0.6.0 | Efficient video reading |
| opencv-python | 4.11.0.86 | Computer vision |
| imageio | 2.37.0 | Image/video I/O |
| imageio-ffmpeg | 0.6.0 | FFmpeg wrapper |
| librosa | 0.11.0 | Audio analysis |
| soundfile | 0.13.1 | Audio file I/O |
| openai-whisper | 20250625 | Speech-to-text |

### ModelScope Ecosystem
| Package | Version | Purpose |
|---------|---------|---------|
| modelscope | 1.31.0 | ModelScope framework |
| dashscope | 1.24.7 | Dashscope API |

### Essential Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| einops | 0.8.1 | Tensor operations |
| safetensors | 0.6.2 | Safe tensor serialization |
| huggingface-hub | 0.35.3 | HF model hub |
| omegaconf | 2.3.0 | Configuration management |
| hydra-core | 1.3.2 | Configuration framework |
| numpy | 1.26.4 | Numerical computing |
| pillow | 11.3.0 | Image processing |

### Supporting Libraries
- `tiktoken==0.12.0` - Token encoding
- `tokenizers==0.21.4` - Fast tokenization
- `ftfy==6.3.1` - Text fixing
- `regex==2025.10.23` - Regular expressions
- `requests==2.32.5` - HTTP library
- `tqdm==4.67.1` - Progress bars
- `matplotlib==3.10.7` - Plotting
- `scipy==1.15.3` - Scientific computing
- `scikit-learn==1.7.2` - Machine learning utilities

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.1 support
- **VRAM**: Minimum 24GB recommended (for 14B models)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and cache

### Software
- **OS**: Linux (recommended), Windows with WSL2, or macOS
- **Python**: 3.10.x
- **CUDA**: 12.1
- **cuDNN**: Compatible with CUDA 12.1

## Verification

### Check Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

### Test GPU
```bash
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Troubleshooting

### Flash Attention Installation Issues
If `flash-attn` fails to install:
```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

### CUDA Compatibility
Ensure your NVIDIA driver supports CUDA 12.1:
```bash
nvidia-smi
```

### Memory Issues
For large models, consider:
- Using model offloading
- Reducing batch size
- Using gradient checkpointing
- Enabling CPU offload in generation scripts

## Directory Structure After Setup

```
your-new-location/
├── venv_wan22/          # Virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
└── Wan2.2/              # Clone the repository here
    ├── wan/
    ├── examples/
    ├── requirements*.txt
    └── ...
```

## Additional Notes

- The virtual environment uses **Python 3.10** specifically
- All NVIDIA CUDA libraries are automatically installed with PyTorch CUDA build
- For multi-GPU setup, refer to `RUNPOD_MULTI_GPU_INSTALLATION_GUIDE.md`
- For speech-to-video specific setup, see `S2V_PARAMETERS.md`

## Quick Start After Setup

1. Activate the environment:
   ```bash
   source venv_wan22/bin/activate
   ```

2. Run a test:
   ```bash
   python -c "from wan import *; print('Setup successful!')"
   ```

3. Check available scripts:
   - `run_s2v_single_gpu.sh` - Single GPU inference
   - `run_s2v_multi_gpu.sh` - Multi-GPU inference
   - `generate.py` - Main generation script

## Support

For issues specific to this setup, refer to:
- `README.md` - Main project documentation
- `INSTALL.md` - Installation guide
- `QUICK_START.md` - Quick start guide
- `TECHNICAL_ANALYSIS.md` - Technical details

---

**Created**: November 22, 2025  
**Environment**: venv_wan22 @ Wan2.2 Project
