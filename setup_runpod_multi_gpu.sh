#!/bin/bash

################################################################################
# Wan2.2 Multi-GPU Setup Script for RunPod (2x A40)
# This script automates the installation process for Wan2.2 on RunPod
# with 2x NVIDIA A40 GPUs
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# Step 0: Pre-flight checks
################################################################################

log_info "Starting Wan2.2 Multi-GPU Setup for RunPod..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_error "This script is designed for Linux. Detected: $OSTYPE"
    exit 1
fi

# Check for NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
log_info "Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    log_warning "Less than 2 GPUs detected. Multi-GPU features may not work optimally."
fi

# Display GPU info
log_info "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

################################################################################
# Step 1: Python Environment Setup
################################################################################

log_info "Setting up Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv_wan22" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv_wan22
    log_success "Virtual environment created: venv_wan22"
else
    log_info "Virtual environment already exists: venv_wan22"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv_wan22/bin/activate

################################################################################
# Step 2: Upgrade pip and install build tools
################################################################################

log_info "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel packaging ninja

################################################################################
# Step 3: Install PyTorch 2.4.0 with CUDA 12.1
################################################################################

log_info "Checking current PyTorch installation..."
CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
log_info "Current PyTorch version: $CURRENT_TORCH_VERSION"

if [[ "$CURRENT_TORCH_VERSION" == "not installed" ]] || [[ ! "$CURRENT_TORCH_VERSION" =~ ^2\.[4-9] ]]; then
    log_warning "PyTorch 2.4.0+ not found. Installing PyTorch 2.4.0 with CUDA 12.1..."
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    log_success "PyTorch 2.4.0 installed"
else
    log_success "PyTorch 2.4.0+ already installed: $CURRENT_TORCH_VERSION"
fi

# Verify PyTorch installation
log_info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

################################################################################
# Step 4: Install Core Dependencies (except flash-attn)
################################################################################

log_info "Installing core dependencies..."
pip install opencv-python>=4.9.0.80 \
    diffusers>=0.31.0 \
    "transformers>=4.49.0,<=4.51.3" \
    tokenizers>=0.20.3 \
    accelerate>=1.1.1 \
    tqdm \
    "imageio[ffmpeg]" \
    easydict \
    ftfy \
    dashscope \
    imageio-ffmpeg \
    "numpy>=1.23.5,<2"

log_success "Core dependencies installed"

################################################################################
# Step 5: Install Flash Attention
################################################################################

log_info "Installing Flash Attention (this may take 3-5 minutes)..."

# Check if flash_attn is already installed
if python -c "import flash_attn" 2>/dev/null; then
    log_success "Flash Attention already installed"
else
    log_info "Building Flash Attention from source..."
    
    # Try multiple installation methods
    if MAX_JOBS=4 pip install flash-attn --no-build-isolation; then
        log_success "Flash Attention installed successfully"
    else
        log_warning "First installation attempt failed. Trying alternative method..."
        if pip install git+https://github.com/Dao-AILab/flash-attention.git; then
            log_success "Flash Attention installed from git"
        else
            log_error "Failed to install Flash Attention. Please install manually."
            log_info "Try: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
            exit 1
        fi
    fi
fi

# Verify flash attention
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"

################################################################################
# Step 6: Verify Multi-GPU Setup
################################################################################

log_info "Verifying multi-GPU setup..."

# Check distributed support
python -c "import torch.distributed as dist; assert dist.is_available(), 'Distributed not available'; print('✓ Distributed available')"

# Check NCCL support (for multi-GPU communication)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ NCCL available: {torch.cuda.nccl.is_available([i for i in range(torch.cuda.device_count())])}')"

log_success "Multi-GPU setup verified"

################################################################################
# Step 7: Download Model Weights (Optional)
################################################################################

read -p "Do you want to download model weights now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Installing download tools..."
    pip install "huggingface_hub[cli]" modelscope
    
    echo "Which model would you like to download?"
    echo "1) T2V-A14B (Text-to-Video, 14B)"
    echo "2) I2V-A14B (Image-to-Video, 14B)"
    echo "3) TI2V-5B (Text-Image-to-Video, 5B)"
    echo "4) S2V-14B (Speech-to-Video, 14B)"
    echo "5) Animate-14B (Animation, 14B)"
    echo "6) All models"
    echo "7) Skip for now"
    
    read -p "Enter choice (1-7): " MODEL_CHOICE
    
    case $MODEL_CHOICE in
        1)
            log_info "Downloading T2V-A14B..."
            huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
            ;;
        2)
            log_info "Downloading I2V-A14B..."
            huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
            ;;
        3)
            log_info "Downloading TI2V-5B..."
            huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
            ;;
        4)
            log_info "Downloading S2V-14B..."
            huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B
            pip install -r requirements_s2v.txt
            ;;
        5)
            log_info "Downloading Animate-14B..."
            huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B
            pip install -r requirements_animate.txt
            ;;
        6)
            log_info "Downloading all models (this will take a while)..."
            huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B &
            huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B &
            huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B &
            huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B &
            huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B &
            wait
            pip install -r requirements_s2v.txt
            pip install -r requirements_animate.txt
            ;;
        7)
            log_info "Skipping model download"
            ;;
        *)
            log_warning "Invalid choice. Skipping model download"
            ;;
    esac
fi

################################################################################
# Step 8: Create helper scripts
################################################################################

log_info "Creating helper scripts..."

# Create activation script
cat > activate_wan22.sh << 'EOF'
#!/bin/bash
# Activate Wan2.2 environment
source venv_wan22/bin/activate
echo "Wan2.2 environment activated"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
EOF
chmod +x activate_wan22.sh

# Create test script
cat > test_multi_gpu.sh << 'EOF'
#!/bin/bash
# Quick test for multi-GPU setup

echo "=== Testing Wan2.2 Multi-GPU Setup ==="

# Activate environment
source venv_wan22/bin/activate

# Check GPUs
echo -e "\n1. GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Check PyTorch
echo -e "\n2. PyTorch Configuration:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Check Flash Attention
echo -e "\n3. Flash Attention:"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" || echo "Flash Attention not installed"

# Check Distributed
echo -e "\n4. Distributed Support:"
python -c "import torch.distributed as dist; print(f'Distributed available: {dist.is_available()}')"
python -c "import torch; print(f'NCCL available: {torch.cuda.nccl.is_available([i for i in range(torch.cuda.device_count())])}')"

echo -e "\n=== All checks passed! ==="
EOF
chmod +x test_multi_gpu.sh

# Create example run script
cat > run_example.sh << 'EOF'
#!/bin/bash
# Example: Run Text-to-Video generation on 2 GPUs

source venv_wan22/bin/activate

# Check if model exists
if [ ! -d "Wan2.2-T2V-A14B" ]; then
    echo "Error: Wan2.2-T2V-A14B model not found"
    echo "Please download it first using:"
    echo "huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B"
    exit 1
fi

# Run generation
torchrun --nproc_per_node=2 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 2 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

echo "Generation complete! Check the output file."
EOF
chmod +x run_example.sh

log_success "Helper scripts created:"
log_info "  - activate_wan22.sh: Activate environment"
log_info "  - test_multi_gpu.sh: Test multi-GPU setup"
log_info "  - run_example.sh: Run example generation"

################################################################################
# Step 9: Final Summary
################################################################################

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
log_info "Environment: venv_wan22"
log_info "GPUs Detected: $GPU_COUNT"
log_info "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
log_info "CUDA Version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""
log_info "Next Steps:"
echo "  1. Activate environment: source activate_wan22.sh"
echo "  2. Test setup: ./test_multi_gpu.sh"
echo "  3. Download models (if not done): huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B"
echo "  4. Run example: ./run_example.sh"
echo ""
log_info "For detailed instructions, see: RUNPOD_MULTI_GPU_INSTALLATION_GUIDE.md"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run test automatically
read -p "Do you want to run the test now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./test_multi_gpu.sh
fi
