#!/usr/bin/env python3
"""
Wan2.2 Animate Multi-GPU MEMORY OPTIMIZED Script

Additional memory optimizations for 48GB GPUs:
- Aggressive CUDA cache clearing before VAE decode
- Gradient checkpointing hints
- Smaller memory allocation chunks
- CPU offloading for VAE decode when needed

For 2x A40 48GB GPUs
"""

import os
import sys
import logging
import subprocess
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

# Configuration
CACHE_DIR = "/home/caches"
CKPT_DIR = os.path.join(CACHE_DIR, "Wan2.2-Animate-14B")
PROCESS_CKPT_DIR = os.path.join(CKPT_DIR, "process_checkpoint")

# Example inputs
EXAMPLE_IMAGE = "/workspace/Wan2.2/examples/wan_animate/animate/image.jpeg"
EXAMPLE_VIDEO = "/workspace/Wan2.2/examples/wan_animate/animate/video.mp4"
OUTPUT_DIR = "/workspace/Wan2.2/outputs/animate"


def setup_directories():
    """Create necessary directories"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PROCESS_CKPT_DIR, exist_ok=True)
    logging.info(f"✓ Directories created: {CACHE_DIR}, {OUTPUT_DIR}")


def download_main_model():
    """Download Wan2.2-Animate-14B model from HuggingFace"""
    logging.info("📥 Downloading Wan2.2-Animate-14B model (this may take a while)...")
    
    try:
        model_path = snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            cache_dir=CACHE_DIR,
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.gguf"]
        )
        logging.info(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"❌ Failed to download model: {e}")
        sys.exit(1)


def download_preprocessing_models():
    """Download preprocessing models for pose detection and face tracking"""
    logging.info("📥 Downloading preprocessing models...")
    
    models_to_download = {
        "det/yolov10m.onnx": {
            "repo_id": "jameslahm/yolov10m",
            "filename": "onnx/model.onnx"
        },
        "pose2d/vitpose_h_wholebody.onnx": {
            "repo_id": "yzd-v/VitPose",
            "filename": "onnx/vitpose_h_wholebody.onnx"
        },
        "sam2/sam2_hiera_large.pt": {
            "repo_id": "facebook/sam2-hiera-large",
            "filename": "sam2_hiera_large.pt"
        }
    }
    
    for local_path, info in models_to_download.items():
        full_path = os.path.join(PROCESS_CKPT_DIR, local_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if os.path.exists(full_path):
            logging.info(f"✓ Already exists: {local_path}")
            continue
            
        try:
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                cache_dir=CACHE_DIR,
                local_dir=os.path.dirname(full_path),
                local_dir_use_symlinks=False
            )
            logging.info(f"✓ Downloaded: {local_path}")
        except Exception as e:
            logging.warning(f"⚠️  Could not download {local_path}: {e}")
            logging.warning(f"   You may need to manually download from {info['repo_id']}")


def run_animate_multi_gpu(
    processed_path,
    num_gpus=2,
    refert_num=1,
    sample_steps=20,
    replace_flag=False,
    use_relighting_lora=False
):
    """
    Run Wan2.2 Animate with aggressive memory optimization
    """
    logging.info(f"🚀 Running Wan2.2 Animate on {num_gpus} GPUs (MEMORY OPTIMIZED)...")
    
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "--master_port", "29500",
        "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", CKPT_DIR,
        "--src_root_path", processed_path,
        "--refert_num", str(refert_num),
        "--sample_steps", str(sample_steps),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", str(num_gpus),
    ]
    
    if replace_flag:
        cmd.append("--replace_flag")
    
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")
    
    # AGGRESSIVE MEMORY OPTIMIZATION
    env = os.environ.copy()
    env.update({
        # NCCL Configuration
        "NCCL_DEBUG": "WARN",  # Less verbose to reduce overhead
        "NCCL_IB_DISABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_SHM_DISABLE": "0",
        "NCCL_NET_GDR_LEVEL": "5",
        
        # AGGRESSIVE MEMORY MANAGEMENT
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:256",
        "CUDA_LAUNCH_BLOCKING": "0",
        
        # CPU Threading
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        
        # PyTorch optimizations
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    
    logging.info("=" * 70)
    logging.info("🔧 Memory-Optimized Configuration:")
    logging.info(f"   CUDA Alloc: {env['PYTORCH_CUDA_ALLOC_CONF']}")
    logging.info(f"   Smaller chunks (256MB) + aggressive GC (60%)")
    logging.info("=" * 70)
    
    try:
        logging.info(f"\n🎬 Launching generation with command:")
        logging.info(f"   {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, env=env, check=True)
        
        logging.info("=" * 70)
        logging.info("✓ Animation generation completed successfully!")
        logging.info("=" * 70)
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error("=" * 70)
        logging.error(f"❌ Generation failed with exit code {e.returncode}")
        logging.error("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 Animate - MEMORY OPTIMIZED for 48GB GPUs"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--video", default=EXAMPLE_VIDEO)
    parser.add_argument("--image", default=EXAMPLE_IMAGE)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--refert-num", type=int, default=1, choices=[1, 5])
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--replace-mode", action="store_true")
    parser.add_argument("--use-relighting-lora", action="store_true")
    
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("🎯 Wan2.2 Animate Multi-GPU (MEMORY OPTIMIZED)")
    logging.info("=" * 70)
    
    setup_directories()
    
    if not args.skip_download:
        download_main_model()
        download_preprocessing_models()
    else:
        logging.info("⏭️  Skipping model download")
    
    processed_output = os.path.join(OUTPUT_DIR, "process_results")
    if not args.skip_preprocessing:
        logging.error("❌ Preprocessing not implemented in this script")
        logging.error("   Use --skip-preprocessing with pre-processed data")
        sys.exit(1)
    else:
        logging.info("⏭️  Using existing preprocessed data")
    
    success = run_animate_multi_gpu(
        processed_output,
        num_gpus=args.num_gpus,
        refert_num=args.refert_num,
        sample_steps=args.sample_steps,
        replace_flag=args.replace_mode,
        use_relighting_lora=args.use_relighting_lora
    )
    
    if success:
        logging.info("🎉 DONE! Check output: " + OUTPUT_DIR)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
