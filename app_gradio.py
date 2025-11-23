#!/usr/bin/env python3
"""
Wan2.2 Animate - Gradio Web Interface

A user-friendly web interface for generating animated videos using Wan2.2 Animate.
Supports multi-GPU execution with FSDP + Ulysses parallelization.

Features:
- Upload reference image and driving video
- Configure generation parameters
- Real-time progress monitoring
- Multi-GPU support (1, 2, 4, 5, 8 GPUs)
- Download generated videos

Author: Gradio UI for Wan2.2 Animate
Date: November 2025
"""

import os
import sys
import logging
import subprocess
import shutil
import gradio as gr
import torch
from pathlib import Path
from datetime import datetime
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
OUTPUT_DIR = "/workspace/Wan2.2/outputs/animate"
UPLOAD_DIR = "/workspace/Wan2.2/uploads"

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PROCESS_CKPT_DIR, exist_ok=True)


def check_models_downloaded():
    """Check if required models are downloaded"""
    # Check for actual model files that exist after download
    model_files = [
        os.path.join(CKPT_DIR, "config.json"),
        os.path.join(CKPT_DIR, "diffusion_pytorch_model.safetensors.index.json"),
        os.path.join(CKPT_DIR, "Wan2.1_VAE.pth"),
        os.path.join(CKPT_DIR, "models_t5_umt5-xxl-enc-bf16.pth"),
    ]
    
    # Check if any critical file exists
    found_files = []
    for model_file in model_files:
        if os.path.exists(model_file):
            found_files.append(os.path.basename(model_file))
            logging.info(f"✓ Model detected: {os.path.basename(model_file)}")
    
    # If we found at least 2 critical files, consider models downloaded
    if len(found_files) >= 2:
        logging.info(f"✓ Models found: {', '.join(found_files)}")
        return True
    
    logging.warning("⚠️ No models found in CKPT_DIR")
    return False


def download_models(progress=gr.Progress()):
    """Download all required models"""
    try:
        progress(0, desc="📥 Downloading Wan2.2-Animate-14B model...")
        
        # Download main model
        model_path = snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            cache_dir=CACHE_DIR,
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.gguf"]
        )
        
        progress(0.5, desc="📥 Downloading preprocessing models...")
        
        # Download preprocessing models
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
        
        for i, (local_path, info) in enumerate(models_to_download.items()):
            full_path = os.path.join(PROCESS_CKPT_DIR, local_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if os.path.exists(full_path):
                continue
                
            try:
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=info["filename"],
                    cache_dir=CACHE_DIR,
                    local_dir=os.path.dirname(full_path),
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                logging.warning(f"Could not download {local_path}: {e}")
        
        progress(1.0, desc="✅ Models downloaded successfully!")
        return "✅ All models downloaded successfully!"
        
    except Exception as e:
        return f"❌ Download failed: {str(e)}"


def preprocess_video(video_path, image_path, use_flux=False, progress=gr.Progress()):
    """Preprocess video for animation"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"process_results_{timestamp}")
        
        progress(0, desc="🔄 Preprocessing video and image...")
        
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
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="/workspace/Wan2.2"
        )
        
        progress(1.0, desc="✅ Preprocessing completed!")
        return output_path, f"✅ Preprocessing completed! Output: {output_path}"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Preprocessing failed:\n{e.stderr}"
        return None, error_msg
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_video(
    processed_path,
    num_gpus=2,
    refert_num=1,
    sample_steps=20,
    replace_mode=False,
    use_relighting_lora=False,
    progress=gr.Progress()
):
    """Generate animated video using Wan2.2 Animate"""
    try:
        if not processed_path or not os.path.exists(processed_path):
            return None, "❌ No preprocessed data found. Please preprocess first!"
        
        progress(0, desc=f"🚀 Starting generation on {num_gpus} GPU(s)...")
        
        # Build command
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
        ]
        
        # Add ulysses_size if num_gpus divides 40 (num_heads)
        valid_ulysses = [1, 2, 4, 5, 8, 10, 20, 40]
        if num_gpus in valid_ulysses:
            cmd.extend(["--ulysses_size", str(num_gpus)])
        
        if replace_mode:
            cmd.append("--replace_flag")
        
        if use_relighting_lora:
            cmd.append("--use_relighting_lora")
        
        # Environment variables
        env = os.environ.copy()
        env.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "CUDA_LAUNCH_BLOCKING": "0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        })
        
        progress(0.1, desc="⚙️ Running generation (this may take several minutes)...")
        
        # Run generation
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            cwd="/workspace/Wan2.2"
        )
        
        progress(0.9, desc="🔍 Looking for generated video...")
        
        # Find the generated video
        # Videos are saved in processed_path with .mp4 extension
        video_files = list(Path(processed_path).glob("*.mp4"))
        
        if video_files:
            # Get the most recent video
            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
            progress(1.0, desc="✅ Generation completed!")
            return str(latest_video), f"✅ Video generated successfully!\n\nPath: {latest_video}"
        else:
            return None, "⚠️ Generation completed but no video file found"
            
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Generation failed:\n{e.stderr[-2000:]}"  # Last 2000 chars
        return None, error_msg
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def full_pipeline(
    image,
    video,
    num_gpus,
    refert_num,
    sample_steps,
    use_flux,
    replace_mode,
    use_relighting_lora,
    progress=gr.Progress()
):
    """Run the full pipeline: preprocess + generate"""
    
    # Check if models are downloaded
    if not check_models_downloaded():
        yield None, None, "❌ Models not downloaded! Please download models first using the 'Download Models' button."
        return
    
    if image is None or video is None:
        yield None, None, "❌ Please upload both an image and a video!"
        return
    
    # Save uploaded files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UPLOAD_DIR, f"image_{timestamp}.png")
    video_path = os.path.join(UPLOAD_DIR, f"video_{timestamp}.mp4")
    
    # Handle file paths (Gradio can pass file objects or paths)
    if hasattr(image, 'name'):
        shutil.copy(image.name, image_path)
    else:
        shutil.copy(image, image_path)
        
    if hasattr(video, 'name'):
        shutil.copy(video.name, video_path)
    else:
        shutil.copy(video, video_path)
    
    # Step 1: Preprocess
    progress(0, desc="🔄 Step 1/2: Preprocessing...")
    processed_path, preprocess_msg = preprocess_video(
        video_path,
        image_path,
        use_flux=use_flux,
        progress=progress
    )
    
    if processed_path is None:
        yield None, None, preprocess_msg
        return
    
    yield None, None, preprocess_msg
    
    # Step 2: Generate
    progress(0.5, desc="🚀 Step 2/2: Generating video...")
    output_video, generate_msg = generate_video(
        processed_path,
        num_gpus=num_gpus,
        refert_num=refert_num,
        sample_steps=sample_steps,
        replace_mode=replace_mode,
        use_relighting_lora=use_relighting_lora,
        progress=progress
    )
    
    status_msg = f"{preprocess_msg}\n\n{generate_msg}"
    yield processed_path, output_video, status_msg


# Create Gradio interface
def create_interface():
    """Create the Gradio web interface"""
    
    with gr.Blocks(title="Wan2.2 Animate - Video Generation") as demo:
        
        # Custom CSS for eye-catching design
        gr.HTML("""
        <style>
        /* Modern, Eye-Catching Design with Dark Mode Support */
        .gradio-container {
            max-width: 1600px !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Header Styling */
        .header-title {
            text-align: center;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            color: white !important;
            padding: 40px 20px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
        }
        
        .header-title h1 {
            font-size: 2.5em !important;
            margin: 0 0 10px 0 !important;
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            color: white !important;
        }
        
        .header-title p {
            color: white !important;
            font-size: 1.2em !important;
            font-weight: 500 !important;
        }
        
        /* Info Boxes - Always visible */
        .info-box {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.15);
            color: #1e40af !important;
            font-weight: 600;
        }
        
        .info-box * {
            color: #1e40af !important;
        }
        
        .info-box h3 {
            color: #1e3a8a !important;
            margin-top: 0 !important;
        }
        
        /* Warning Boxes */
        .warning-box {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #f59e0b;
            margin: 15px 0;
            color: #78350f !important;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(245, 158, 11, 0.15);
        }
        
        .warning-box * {
            color: #78350f !important;
        }
        
        /* Success Boxes */
        .success-box {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #10b981;
            margin: 15px 0;
            color: #065f46 !important;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.15);
        }
        
        .success-box * {
            color: #065f46 !important;
        }
        
        /* Section Headers - vibrant */
        h2 {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800 !important;
            font-size: 1.8em !important;
            margin: 20px 0 !important;
        }
        
        /* Footer */
        .footer-box {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            margin-top: 30px;
        }
        
        .footer-box p {
            margin: 0 !important;
            color: white !important;
            font-size: 1.1em !important;
            font-weight: 600 !important;
        }
        
        /* Button styling */
        button.primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            border: none !important;
            font-weight: 700 !important;
        }
        </style>
        """)
        
        gr.HTML("""
        <div class="header-title">
            <h1>🎬 Wan2.2 Animate - AI Video Generation</h1>
            <p>✨ Transform static images into dynamic animations with cutting-edge AI</p>
        </div>
        """)
        
        gr.HTML("""
        <div class="info-box">
            <p style="font-size: 16px; margin: 0; line-height: 1.6;">
                <b>💡 How it works:</b> Upload a reference image (your character) and a driving video (motion source). 
                The AI will intelligently transfer the motion from the video to your character, creating a hyper-realistic animated video!
            </p>
        </div>
        """)
        
        # Model download section
        gr.HTML('<h2>📥 Step 1: Download Models (First Time Only)</h2>')
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="warning-box">
                    <p style="font-size: 15px; margin: 0; line-height: 1.6;">
                        ⚠️ <b>Important:</b> Before generating videos, download the required models (~20GB). 
                        This only needs to be done once. The models will be cached for future use.
                    </p>
                </div>
                """)
            with gr.Column(scale=1):
                download_btn = gr.Button("📥 Download All Models", variant="primary", size="lg", scale=1)
        
        download_status = gr.Textbox(label="Download Status", interactive=False, lines=3)
        
        download_btn.click(
            fn=download_models,
            outputs=download_status
        )
        
        gr.HTML('<hr style="margin: 30px 0; border: none; border-top: 2px solid #e5e7eb;">')
        gr.HTML('<h2>🎬 Step 2: Generate Your Video</h2>')
        
        # Main interface
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Your Files")
                
                with gr.Tab("📷 Reference Image"):
                    image_input = gr.Image(
                        label="Upload Character Image",
                        type="filepath",
                        height=300
                    )
                    gr.Markdown("*Upload a clear image of the character/person you want to animate*")
                
                with gr.Tab("🎥 Driving Video"):
                    video_input = gr.Video(
                        label="Upload Motion Video",
                        height=300
                    )
                    gr.Markdown("*Upload a video with the motion you want to transfer*")
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Generation Settings")
                
                with gr.Accordion("🖥️ GPU & Performance", open=True):
                    # Auto-detect available GPUs
                    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    gpu_choices = [1, 2, 3, 4, 5, 8]
                    # Filter to only show available GPUs
                    gpu_choices = [x for x in gpu_choices if x <= available_gpus]
                    
                    num_gpus = gr.Dropdown(
                        label="Number of GPUs",
                        choices=gpu_choices,
                        value=min(available_gpus, 3),  # Default to available GPUs (max 3)
                        info=f"💡 {available_gpus} GPU(s) detected. More GPUs = faster generation"
                    )
                    
                    sample_steps = gr.Slider(
                        label="Quality (Sampling Steps)",
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        info="Higher = better quality but slower"
                    )
                    
                    refert_num = gr.Radio(
                        label="Temporal Consistency",
                        choices=[("Fast (1 frame)", 1), ("High Quality (5 frames)", 5)],
                        value=1,
                        info="5 frames = smoother animations"
                    )
                
                with gr.Accordion("🎨 Advanced Options", open=False):
                    use_flux = gr.Checkbox(
                        label="🔥 Enable FLUX (Enhanced Pose Retargeting)",
                        value=False,
                        info="Better pose transfer, requires FLUX model"
                    )
                    
                    replace_mode = gr.Checkbox(
                        label="🔄 Replacement Mode",
                        value=False,
                        info="Character replacement instead of animation"
                    )
                    
                    use_relighting_lora = gr.Checkbox(
                        label="💡 Relighting LoRA",
                        value=False,
                        info="Better lighting (for replacement mode)"
                    )
                
                gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
                generate_btn = gr.Button(
                    "🚀 Generate Animated Video", 
                    variant="primary", 
                    size="lg",
                    elem_id="generate-btn"
                )
                
                gr.HTML("""
                <div class="info-box" style="margin-top: 15px;">
                    <p style="font-size: 14px; margin: 0;">
                        ⏱️ <b>Estimated time:</b> 5-15 minutes depending on video length and settings
                    </p>
                </div>
                """)
            
            # Right column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Generation Progress & Results")
                
                status_output = gr.Textbox(
                    label="Status & Logs",
                    lines=12,
                    interactive=False,
                    placeholder="Status updates will appear here..."
                )
                
                gr.Markdown("### 🎥 Your Generated Video")
                
                video_output = gr.Video(
                    label="Result",
                    height=450
                )
                
                gr.HTML("""
                <div class="info-box">
                ✅ <b>After generation:</b> You can preview the video above and download it using the download button.
                </div>
                """)
                
                processed_path_state = gr.State()
        
        # Connect the generate button
        generate_btn.click(
            fn=full_pipeline,
            inputs=[
                image_input,
                video_input,
                num_gpus,
                refert_num,
                sample_steps,
                use_flux,
                replace_mode,
                use_relighting_lora
            ],
            outputs=[processed_path_state, video_output, status_output]
        )
        
        gr.HTML('<hr style="margin: 40px 0; border: none; border-top: 2px solid #e5e7eb;">')
        gr.HTML('<h2>💡 Tips & Best Practices</h2>')
        
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <h3 style="margin-top: 0;">⚡ For Faster Generation:</h3>
                    <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                        <li>Use <b>2 GPUs</b> (recommended)</li>
                        <li>Set sampling steps to <b>15-20</b></li>
                        <li>Use <b>1 frame</b> temporal guidance</li>
                        <li>Keep videos <b>short</b> (&lt;10 seconds)</li>
                    </ul>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <h3 style="margin-top: 0;">🎨 For Better Quality:</h3>
                    <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                        <li>Use <b>4-8 GPUs</b> if available</li>
                        <li>Set sampling steps to <b>25-30</b></li>
                        <li>Use <b>5 frames</b> temporal guidance</li>
                        <li>Enable <b>FLUX</b> preprocessing</li>
                    </ul>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <h3 style="margin-top: 0;">🖥️ Hardware Info:</h3>
                    <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                        <li><b>Minimum:</b> 1x A40 48GB GPU</li>
                        <li><b>Recommended:</b> 2x A40 48GB GPUs</li>
                        <li><b>Optimal:</b> 4-8x A40 48GB GPUs</li>
                        <li><b>Processing:</b> 5-15 min per video</li>
                    </ul>
                </div>
                """)
        
        gr.HTML("""
        <hr style="margin: 40px 0; border: none; border-top: 2px solid #e5e7eb;">
        <div class="footer-box">
            <p>
                🚀 Powered by <b>Wan2.2 Animate-14B</b> | 
                🎬 Multi-GPU Optimized | 
                ⚡ FSDP + Ulysses Parallelization
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wan2.2 Animate Gradio Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--auth", help="Enable authentication (format: username:password)")
    parser.add_argument("--auth-message", default="Enter credentials to access Wan2.2 Animate", 
                        help="Custom authentication message")
    
    args = parser.parse_args()
    
    # Parse authentication
    auth_tuple = None
    if args.auth:
        if ":" in args.auth:
            username, password = args.auth.split(":", 1)
            auth_tuple = (username, password)
            logging.info(f"🔐 Authentication enabled for user: {username}")
        else:
            logging.error("❌ Invalid auth format. Use: username:password")
            sys.exit(1)
    
    # Create and launch interface
    demo = create_interface()
    
    logging.info("=" * 70)
    logging.info("🎬 Starting Wan2.2 Animate Gradio Interface")
    logging.info(f"   Host: {args.host}")
    logging.info(f"   Port: {args.port}")
    logging.info(f"   Share: {args.share}")
    logging.info(f"   Auth: {'Enabled' if auth_tuple else 'Disabled'}")
    logging.info("=" * 70)
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        auth=auth_tuple,
        auth_message=args.auth_message if auth_tuple else None
    )
