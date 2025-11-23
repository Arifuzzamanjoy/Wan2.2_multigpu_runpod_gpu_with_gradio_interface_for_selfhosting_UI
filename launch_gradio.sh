#!/bin/bash
# Launch script for Wan2.2 Animate Gradio Interface

echo "🎬 Launching Wan2.2 Animate Gradio Interface"
echo "============================================="

# Activate virtual environment
source /workspace/Wan2.2/venv_wan22/bin/activate

# Change to workspace directory
cd /workspace/Wan2.2

# Check for authentication
if [ -z "$WAN_AUTH" ]; then
    echo "ℹ️  No authentication set (public access)"
    echo "   To enable auth, set: export WAN_AUTH='username:password'"
    python app_gradio.py --host 0.0.0.0 --port 7860 --share
else
    echo "🔐 Authentication enabled"
    python app_gradio.py --host 0.0.0.0 --port 7860 --share --auth "$WAN_AUTH"
fi

# Usage examples:
# 1. Public access (no auth):
#    ./launch_gradio.sh
#
# 2. With authentication:
#    export WAN_AUTH="admin:your_secure_password"
#    ./launch_gradio.sh
#
# 3. Custom port:
#    python app_gradio.py --port 8080
#
# 4. Local only (no share link):
#    python app_gradio.py --host 127.0.0.1 --port 7860
