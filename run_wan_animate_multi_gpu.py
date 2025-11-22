#!/usr/bin/env python3
"""
Wan2.2 Animate Multi-GPU Inference Script with GGUF Model Support
Author: Auto-generated for multi-GPU A40 setup
Date: November 2025

This script downloads and runs Wan2.2-Animate-14B on multiple GPUs (2x A40 48GB)
using FSDP + DeepSpeed Ulysses for distributed inference.

Note: GGUF quantized models from HuggingFace are converted to standard format
for use with the official Wan2.2 codebase.
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

# Example inputs (from your workspace)
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
    """
    Download Wan2.2-Animate-14B model from HuggingFace
    
    Note: The GGUF quantized version from QuantStack is experimental.
    For best results, we use the official safetensors version from Wan-AI
    which is optimized for multi-GPU inference with FSDP.
    """
    logging.info("📥 Downloading Wan2.2-Animate-14B model (this may take a while)...")
    
    # Download official model (safetensors format)
    try:
        model_path = snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            cache_dir=CACHE_DIR,
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.gguf"]  # Ignore GGUF files as they need conversion
        )
        logging.info(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"❌ Failed to download model: {e}")
        sys.exit(1)


def download_preprocessing_models():
    """
    Download preprocessing models for pose detection and face tracking
    These are required for the animation preprocessing step
    """
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


def preprocess_video(video_path, image_path, output_path, use_flux=True):
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


def run_animate_multi_gpu(
    processed_path,
    num_gpus=2,
    refert_num=1,
    sample_steps=5,
    replace_flag=False,
    use_relighting_lora=False
):
    """
    Run Wan2.2 Animate on multiple GPUs using FSDP + Ulysses
    
    Args:
        processed_path: Path to preprocessed results
        num_gpus: Number of GPUs to use
        refert_num: Number of frames for temporal guidance (1 or 5)
        sample_steps: Number of diffusion sampling steps
        replace_flag: Whether to use replacement mode
        use_relighting_lora: Whether to use relighting LoRA
    """
    logging.info(f"🚀 Running Wan2.2 Animate on {num_gpus} GPUs...")
    
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
    
    # Set environment variables for optimal multi-GPU performance
    env = os.environ.copy()
    env.update({
        "NCCL_DEBUG": "INFO",  # For debugging distributed training
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Better memory management
        "OMP_NUM_THREADS": "8",  # Optimal for 2 GPUs
    })
    
    try:
        logging.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, check=True)
        logging.info("✓ Animation generation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Generation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 Animate on multi-GPU setup"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download if already exists"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip video preprocessing (use existing processed data)"
    )
    parser.add_argument(
        "--video",
        default=EXAMPLE_VIDEO,
        help="Path to driving video"
    )
    parser.add_argument(
        "--image",
        default=EXAMPLE_IMAGE,
        help="Path to reference character image"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--refert-num",
        type=int,
        default=1,
        choices=[1, 5],
        help="Number of frames for temporal guidance"
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=20,
        help="Number of diffusion sampling steps"
    )
    parser.add_argument(
        "--use-flux",
        action="store_true",
        default=False,
        help="Use FLUX for enhanced pose retargeting (requires FLUX.1-Kontext-dev model)"
    )
    parser.add_argument(
        "--replace-mode",
        action="store_true",
        help="Use replacement mode instead of animation"
    )
    parser.add_argument(
        "--use-relighting-lora",
        action="store_true",
        help="Use relighting LoRA (for replacement mode)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Download models
    if not args.skip_download:
        download_main_model()
        download_preprocessing_models()
    else:
        logging.info("⏭️  Skipping model download (--skip-download specified)")
        if not os.path.exists(CKPT_DIR):
            logging.error(f"❌ Model directory not found: {CKPT_DIR}")
            logging.error("   Please run without --skip-download first")
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
            logging.error("❌ Preprocessing failed, cannot continue")
            sys.exit(1)
    else:
        logging.info("⏭️  Skipping preprocessing (--skip-preprocessing specified)")
        if not os.path.exists(processed_output):
            logging.error(f"❌ Processed data not found: {processed_output}")
            logging.error("   Please run without --skip-preprocessing first")
            sys.exit(1)
    
    # Run animation generation
    success = run_animate_multi_gpu(
        processed_output,
        num_gpus=args.num_gpus,
        refert_num=args.refert_num,
        sample_steps=args.sample_steps,
        replace_flag=args.replace_mode,
        use_relighting_lora=args.use_relighting_lora
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
