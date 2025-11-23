#!/bin/bash
# Quick Demo - Wan2.2 Animate Gradio Interface
# This script shows different ways to launch the app

echo "🎬 Wan2.2 Animate - Launch Options Demo"
echo "========================================"
echo ""
echo "Choose how you want to launch:"
echo ""
echo "1. 🌐 Public Access (with default auth: admin/wan2024)"
echo "2. 🔒 Custom Authentication"
echo "3. 🏠 Local Only (no share link, no auth)"
echo "4. ⚡ Quick Start (public, no auth - NOT RECOMMENDED)"
echo ""
read -p "Enter choice (1-4): " choice

# Activate venv
source /workspace/Wan2.2/venv_wan22/bin/activate
cd /workspace/Wan2.2

case $choice in
    1)
        echo ""
        echo "🔐 Launching with default credentials..."
        echo "   Username: admin"
        echo "   Password: wan2024"
        echo ""
        echo "⚠️  CHANGE THESE CREDENTIALS IN PRODUCTION!"
        echo ""
        python app_gradio.py \
            --host 0.0.0.0 \
            --port 7860 \
            --share \
            --auth "admin:wan2024"
        ;;
    2)
        echo ""
        read -p "Enter username: " username
        read -sp "Enter password: " password
        echo ""
        echo ""
        echo "🔐 Launching with custom credentials..."
        python app_gradio.py \
            --host 0.0.0.0 \
            --port 7860 \
            --share \
            --auth "$username:$password"
        ;;
    3)
        echo ""
        echo "🏠 Launching local-only server (no internet access)..."
        echo "   Access at: http://localhost:7860"
        echo ""
        python app_gradio.py \
            --host 127.0.0.1 \
            --port 7860
        ;;
    4)
        echo ""
        echo "⚠️  WARNING: No authentication! Anyone with link can access!"
        read -p "Continue? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python app_gradio.py \
                --host 0.0.0.0 \
                --port 7860 \
                --share
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
