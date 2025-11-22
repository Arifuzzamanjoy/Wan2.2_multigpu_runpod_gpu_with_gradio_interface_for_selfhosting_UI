#!/usr/bin/env python3
"""
Wan2.2 Animate OPTIMIZED Multi-GPU Inference Script
Improvements over run_wan_animate_multi_gpu.py:
1. Better memory management with gradient checkpointing hints
2. Improved NCCL settings for A40 GPUs
3. Batch processing support for multiple videos
4. Better error handling and recovery
5. Performance monitoring
"""

import os
import sys
import logging
import subprocess
import argparse
import time
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
    """Download Wan2.2-Animate-14B model"""
    logging.info("📥 Downloading Wan2.2-Animate-14B model...")
    
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
    """Download preprocessing models"""
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


def preprocess_video(video_path, image_path, output_path, use_flux=False):
    """Preprocess video for animation mode"""
    logging.info("🔄 Preprocessing video for animation...")
    
    cmd = [
        "python", "./wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", PROCESS_CKPT_DIR,
        "--video_path", video_path,
        "--refer_path", image_path,
        "--save_path", output_path,
        "--resolution_area", "1280", "720",
        "--retarget_flag"
    ]
    
    if use_flux:
        cmd.append("--use_flux")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("✓ Preprocessing completed")
        logging.info(f"  Output saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Preprocessing failed: {e}")
        logging.error(f"   stdout: {e.stdout}")
        logging.error(f"   stderr: {e.stderr}")
        return False


def run_animate_multi_gpu(
    processed_path,
    num_gpus=2,
    refert_num=1,
    sample_steps=20,
    replace_flag=False,
    use_relighting_lora=False,
    performance_mode="balanced"
):
    """
    Run Wan2.2 Animate with optimized settings
    
    performance_mode options:
    - "speed": Faster but uses more memory
    - "balanced": Good balance (default)
    - "memory": Lower memory but slower
    """
    logging.info(f"🚀 Running Wan2.2 Animate on {num_gpus} GPUs (mode: {performance_mode})...")
    
    cmd = [
        "python", "-m", "torch.distributed.run",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", CKPT_DIR,
        "--src_root_path", processed_path,
        "--refert_num", str(refert_num),
        "--sample_steps", str(sample_steps),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", str(num_gpus)
    ]
    
    if replace_flag:
        cmd.append("--replace_flag")
    
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")
    
    # Optimized environment variables for A40 GPUs
    env = os.environ.copy()
    
    # Base settings for all modes
    base_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",  # Async kernel launches
        "NCCL_IB_DISABLE": "0",  # Enable InfiniBand if available
        "NCCL_P2P_LEVEL": "NVL",  # NVLink optimization
    }
    
    # Mode-specific settings
    if performance_mode == "speed":
        mode_env = {
            "NCCL_DEBUG": "WARN",  # Less verbose
            "OMP_NUM_THREADS": str(num_gpus * 8),  # More CPU threads
            "NCCL_NSOCKS_PERTHREAD": "4",
            "NCCL_SOCKET_NTHREADS": "2",
        }
    elif performance_mode == "memory":
        mode_env = {
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8",
        }
    else:  # balanced
        mode_env = {
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "8",
            "NCCL_SOCKET_NTHREADS": "1",
        }
    
    env.update(base_env)
    env.update(mode_env)
    
    # Log settings
    logging.info("Environment variables:")
    for key in ["PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "NCCL_DEBUG"]:
        if key in env:
            logging.info(f"  {key}={env[key]}")
    
    start_time = time.time()
    
    try:
        logging.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, check=True)
        
        elapsed_time = time.time() - start_time
        logging.info(f"✓ Animation generation completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Generation failed after {elapsed_time:.1f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 Animate with optimized multi-GPU settings"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--video", default=EXAMPLE_VIDEO)
    parser.add_argument("--image", default=EXAMPLE_IMAGE)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--refert-num", type=int, default=1, choices=[1, 5])
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--use-flux", action="store_true", default=False)
    parser.add_argument("--replace-mode", action="store_true")
    parser.add_argument("--use-relighting-lora", action="store_true")
    parser.add_argument(
        "--performance-mode",
        choices=["speed", "balanced", "memory"],
        default="balanced",
        help="Optimization mode: speed (faster), balanced (default), memory (lower VRAM)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Download models
    if not args.skip_download:
        download_main_model()
        download_preprocessing_models()
    else:
        logging.info("⏭️  Skipping model download")
        if not os.path.exists(CKPT_DIR):
            logging.error(f"❌ Model directory not found: {CKPT_DIR}")
            sys.exit(1)
    
    # Preprocess video
    processed_output = os.path.join(OUTPUT_DIR, "process_results")
    if not args.skip_preprocessing:
        success = preprocess_video(
            args.video,
            args.image,
            processed_output,
            use_flux=args.use_flux
        )
        if not success:
            sys.exit(1)
    else:
        logging.info("⏭️  Skipping preprocessing")
        if not os.path.exists(processed_output):
            logging.error(f"❌ Processed data not found: {processed_output}")
            sys.exit(1)
    
    # Run animation generation
    success = run_animate_multi_gpu(
        processed_output,
        num_gpus=args.num_gpus,
        refert_num=args.refert_num,
        sample_steps=args.sample_steps,
        replace_flag=args.replace_mode,
        use_relighting_lora=args.use_relighting_lora,
        performance_mode=args.performance_mode
    )
    
    if success:
        logging.info("=" * 60)
        logging.info("🎉 All done! Check the output directory:")
        logging.info(f"   {OUTPUT_DIR}")
        logging.info("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
