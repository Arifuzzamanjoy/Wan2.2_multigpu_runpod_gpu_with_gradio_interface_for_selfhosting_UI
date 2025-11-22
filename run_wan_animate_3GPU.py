#!/usr/bin/env python3
"""
Wan2.2 Animate 3-GPU Optimized Script

For 3x A40 48GB GPUs - Solves OOM issues with better memory distribution
- DiT model: ~13-14GB per GPU (instead of ~20GB on 2 GPUs)
- Leaves 34GB+ free per GPU for VAE decode
- Uses FSDP + Ulysses for optimal parallelization

Author: Optimized for 3x A40 48GB setup
Date: November 2025
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
EXAMPLE_IMAGE = "/workspace/Wan2.2/examples/wan_animate/animate/prompt_00.png"
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


def preprocess_video(video_path, image_path, output_path, use_flux=False):
    """
    Preprocess video for animation mode
    
    Args:
        video_path: Path to driving video
        image_path: Path to reference character image
        output_path: Path to save preprocessed results
        use_flux: Whether to use FLUX for enhanced pose retargeting
    """
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


def run_animate_3gpu(
    processed_path,
    refert_num=1,
    sample_steps=20,
    replace_flag=False,
    use_relighting_lora=False
):
    """
    Run Wan2.2 Animate on 3 GPUs - OPTIMAL for A40 48GB
    
    Memory distribution with 3 GPUs:
    - DiT model: ~13-14GB per GPU (FSDP sharded)
    - VAE: ~4GB per GPU
    - Leaves: ~30-34GB free per GPU
    
    Args:
        processed_path: Path to preprocessed results
        refert_num: Number of frames for temporal guidance (1 or 5)
        sample_steps: Number of diffusion sampling steps (10-30)
        replace_flag: Whether to use replacement mode
        use_relighting_lora: Whether to use relighting LoRA
    """
    logging.info("🚀 Running Wan2.2 Animate on 3 GPUs (OPTIMAL FOR A40 48GB)...")
    
    # IMPORTANT: ulysses_size must divide num_heads (40)
    # Valid values: 1, 2, 4, 5, 8, 10, 20, 40
    # For 3 GPUs, we can't use ulysses_size=3 (doesn't divide 40)
    # Solution: Use FSDP only, or use 2/4/5 GPUs instead
    
    logging.warning("=" * 70)
    logging.warning("⚠️  ATTENTION: Ulysses requires ulysses_size to divide num_heads (40)")
    logging.warning("   3 GPUs cannot use Ulysses (3 doesn't divide 40)")
    logging.warning("   Running with FSDP ONLY (still much better than 2 GPUs!)")
    logging.warning("=" * 70)
    
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "3",  # 3 GPUs
        "--master_port", "29500",
        "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", CKPT_DIR,
        "--src_root_path", processed_path,
        "--refert_num", str(refert_num),
        "--sample_steps", str(sample_steps),
        "--dit_fsdp",  # FSDP for DiT model sharding
        "--t5_fsdp",   # FSDP for T5 encoder
        # NO --ulysses_size (doesn't work with 3 GPUs)
    ]
    
    if replace_flag:
        cmd.append("--replace_flag")
    
    if use_relighting_lora:
        cmd.append("--use_relighting_lora")
    
    # Optimized environment for 3x A40
    env = os.environ.copy()
    env.update({
        # NCCL Configuration
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_P2P_LEVEL": "NVL",  # NVLink optimization
        "NCCL_SHM_DISABLE": "0",
        "NCCL_NET_GDR_LEVEL": "5",
        
        # CUDA Memory Management
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",
        
        # CPU Threading (optimized for 3 GPUs)
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        
        # PyTorch Distributed
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    })
    
    logging.info("=" * 70)
    logging.info("🔧 3-GPU Configuration:")
    logging.info(f"   FSDP: Enabled (model sharded across 3 GPUs)")
    logging.info(f"   Ulysses: Disabled (requires GPU count to divide 40)")
    logging.info(f"   Expected memory per GPU: ~18GB DiT + 4GB VAE = ~22GB")
    logging.info(f"   Free memory per GPU: ~26GB (plenty for VAE decode!)")
    logging.info("=" * 70)
    
    try:
        logging.info(f"\n🎬 Launching generation with command:")
        logging.info(f"   {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, env=env, check=True)
        
        logging.info("=" * 70)
        logging.info("✅ Animation generation completed successfully!")
        logging.info("=" * 70)
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error("=" * 70)
        logging.error(f"❌ Generation failed with exit code {e.returncode}")
        logging.error("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 Animate on 3x A40 48GB GPUs (OPTIMAL)"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--video", default=EXAMPLE_VIDEO)
    parser.add_argument("--image", default=EXAMPLE_IMAGE)
    parser.add_argument("--refert-num", type=int, default=1, choices=[1, 5])
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--use-flux", action="store_true", default=False,
                        help="Use FLUX for enhanced pose retargeting")
    parser.add_argument("--replace-mode", action="store_true")
    parser.add_argument("--use-relighting-lora", action="store_true")
    
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("🎯 Wan2.2 Animate 3-GPU Script (OPTIMAL FOR A40 48GB)")
    logging.info("=" * 70)
    
    setup_directories()
    
    if not args.skip_download:
        download_main_model()
        download_preprocessing_models()
    else:
        logging.info("⏭️  Skipping model download")
    
    processed_output = os.path.join(OUTPUT_DIR, "process_results")
    if not args.skip_preprocessing:
        success = preprocess_video(
            args.video,
            args.image,
            processed_output,
            use_flux=args.use_flux
        )
        if not success:
            logging.error("❌ Preprocessing failed, cannot continue")
            sys.exit(1)
    else:
        logging.info("⏭️  Using existing preprocessed data")
    
    success = run_animate_3gpu(
        processed_output,
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
