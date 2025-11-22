#!/bin/bash
# GPU Multi-GPU Setup Diagnostic Script
# Checks if your system is properly configured for FSDP + Ulysses

echo "======================================================================="
echo "🔍 Wan2.2 Multi-GPU Diagnostic Tool"
echo "======================================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counter
PASSED=0
FAILED=0
WARNINGS=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check Python environment
echo "1. Checking Python Environment..."
if [ -d "/workspace/Wan2.2/venv_wan22" ]; then
    check_pass "Virtual environment found: /workspace/Wan2.2/venv_wan22"
else
    check_fail "Virtual environment NOT found: /workspace/Wan2.2/venv_wan22"
fi

# 2. Check GPU availability
echo ""
echo "2. Checking GPU Availability..."
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 2 ]; then
    check_pass "Found 2 GPUs (optimal for this setup)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif [ "$GPU_COUNT" -gt 0 ]; then
    check_warn "Found $GPU_COUNT GPUs (script configured for 2, but will work)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    check_fail "No GPUs found! nvidia-smi not working or no GPU available"
fi

# 3. Check GPU interconnect
echo ""
echo "3. Checking GPU Interconnect..."
if command -v nvidia-smi &> /dev/null; then
    TOPO=$(nvidia-smi topo -m 2>/dev/null)
    if echo "$TOPO" | grep -q "NV"; then
        check_pass "NVLink detected (optimal for multi-GPU)"
        echo "   Recommendation: Use NCCL_P2P_LEVEL=NVL"
    else
        check_warn "No NVLink detected (using PCIe)"
        echo "   Recommendation: Use NCCL_P2P_LEVEL=SYS"
    fi
fi

# 4. Check CUDA version
echo ""
echo "4. Checking CUDA Version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
if [ ! -z "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        check_pass "CUDA $CUDA_VERSION (compatible)"
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        check_warn "CUDA $CUDA_VERSION (works, but 12.x recommended)"
    else
        check_fail "CUDA $CUDA_VERSION (too old, need 11.8+)"
    fi
else
    check_fail "Could not detect CUDA version"
fi

# 5. Check PyTorch installation
echo ""
echo "5. Checking PyTorch Installation..."
if [ -f "/workspace/Wan2.2/venv_wan22/bin/python" ]; then
    TORCH_VERSION=$(/workspace/Wan2.2/venv_wan22/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        check_pass "PyTorch $TORCH_VERSION installed"
        
        # Check CUDA availability in PyTorch
        TORCH_CUDA=$(/workspace/Wan2.2/venv_wan22/bin/python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$TORCH_CUDA" == "True" ]; then
            check_pass "PyTorch CUDA support enabled"
            
            # Check NCCL
            NCCL_VERSION=$(/workspace/Wan2.2/venv_wan22/bin/python -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null)
            if [ $? -eq 0 ]; then
                check_pass "NCCL version: $NCCL_VERSION"
            else
                check_warn "Could not detect NCCL version"
            fi
        else
            check_fail "PyTorch CUDA support NOT available"
        fi
    else
        check_fail "PyTorch not installed in venv"
    fi
else
    check_fail "Python interpreter not found in venv"
fi

# 6. Check model checkpoint
echo ""
echo "6. Checking Model Checkpoints..."
if [ -d "/home/caches/Wan2.2-Animate-14B" ]; then
    MODEL_SIZE=$(du -sh /home/caches/Wan2.2-Animate-14B 2>/dev/null | cut -f1)
    check_pass "Model checkpoint found: /home/caches/Wan2.2-Animate-14B ($MODEL_SIZE)"
    
    # Check for key files
    if [ -f "/home/caches/Wan2.2-Animate-14B/config.json" ]; then
        check_pass "config.json found"
    else
        check_warn "config.json not found (model may be incomplete)"
    fi
else
    check_warn "Model checkpoint NOT found (will download on first run)"
fi

# 7. Check preprocessing models
echo ""
echo "7. Checking Preprocessing Models..."
PREPROCESS_DIR="/home/caches/Wan2.2-Animate-14B/process_checkpoint"
if [ -d "$PREPROCESS_DIR" ]; then
    check_pass "Preprocessing directory exists"
    
    # Check individual models
    if [ -f "$PREPROCESS_DIR/det/yolov10m.onnx" ]; then
        check_pass "YOLOv10 model found"
    else
        check_warn "YOLOv10 model not found"
    fi
    
    if [ -f "$PREPROCESS_DIR/pose2d/vitpose_h_wholebody.onnx" ]; then
        check_pass "VitPose model found"
    else
        check_warn "VitPose model not found"
    fi
    
    if [ -f "$PREPROCESS_DIR/sam2/sam2_hiera_large.pt" ]; then
        check_pass "SAM2 model found"
    else
        check_warn "SAM2 model not found"
    fi
else
    check_warn "Preprocessing models not found (will download on first run)"
fi

# 8. Check disk space
echo ""
echo "8. Checking Disk Space..."
CACHE_AVAIL=$(df -h /home/caches 2>/dev/null | awk 'NR==2 {print $4}')
WORKSPACE_AVAIL=$(df -h /workspace 2>/dev/null | awk 'NR==2 {print $4}')
if [ ! -z "$CACHE_AVAIL" ]; then
    CACHE_GB=$(echo $CACHE_AVAIL | sed 's/G//')
    if [ $(echo "$CACHE_GB > 50" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
        check_pass "Cache disk space: $CACHE_AVAIL available (sufficient)"
    else
        check_warn "Cache disk space: $CACHE_AVAIL available (may need more for models)"
    fi
fi

if [ ! -z "$WORKSPACE_AVAIL" ]; then
    WS_GB=$(echo $WORKSPACE_AVAIL | sed 's/G//')
    if [ $(echo "$WS_GB > 10" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
        check_pass "Workspace disk space: $WORKSPACE_AVAIL available (sufficient)"
    else
        check_warn "Workspace disk space: $WORKSPACE_AVAIL available (low)"
    fi
fi

# 9. Check for running processes
echo ""
echo "9. Checking for Running Processes..."
TORCH_PROCS=$(pgrep -f torchrun | wc -l)
GEN_PROCS=$(pgrep -f generate.py | wc -l)
if [ "$TORCH_PROCS" -eq 0 ] && [ "$GEN_PROCS" -eq 0 ]; then
    check_pass "No conflicting processes running"
else
    check_warn "$TORCH_PROCS torchrun + $GEN_PROCS generate.py processes running"
    echo "   Run 'pkill -f torchrun' to clean up if needed"
fi

# 10. Check example files
echo ""
echo "10. Checking Example Files..."
if [ -f "/workspace/Wan2.2/examples/wan_animate/animate/image.jpeg" ]; then
    check_pass "Example image found"
else
    check_warn "Example image not found"
fi

if [ -f "/workspace/Wan2.2/examples/wan_animate/animate/video.mp4" ]; then
    check_pass "Example video found"
else
    check_warn "Example video not found"
fi

# 11. Test GPU peer-to-peer access
echo ""
echo "11. Testing GPU P2P Access..."
if [ "$GPU_COUNT" -ge 2 ] && [ -f "/workspace/Wan2.2/venv_wan22/bin/python" ]; then
    P2P_TEST=$(/workspace/Wan2.2/venv_wan22/bin/python -c "
import torch
if torch.cuda.device_count() >= 2:
    can_access = torch.cuda.can_device_access_peer(0, 1)
    print('True' if can_access else 'False')
else:
    print('False')
" 2>/dev/null)
    
    if [ "$P2P_TEST" == "True" ]; then
        check_pass "GPU P2P access enabled (optimal)"
    else
        check_warn "GPU P2P access not available (may impact performance)"
    fi
fi

# Summary
echo ""
echo "======================================================================="
echo "📊 Diagnostic Summary"
echo "======================================================================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ System is ready for multi-GPU inference!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv: source /workspace/Wan2.2/venv_wan22/bin/activate"
    echo "  2. Run script: python /workspace/Wan2.2/run_wan_animate_FIXED.py --skip-download"
elif [ "$FAILED" -le 2 ]; then
    echo -e "${YELLOW}⚠️  System has minor issues but may still work${NC}"
    echo "Review failed checks above and fix if possible"
else
    echo -e "${RED}❌ System has critical issues${NC}"
    echo "Please fix the failed checks before proceeding"
fi

echo ""
echo "For detailed multi-GPU guide, see:"
echo "  - /workspace/Wan2.2/MULTI_GPU_METHODS.md"
echo "  - /workspace/Wan2.2/QUICK_START_FIXED.md"
echo "======================================================================="
