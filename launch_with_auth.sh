#!/bin/bash
# Launch Wan2.2 Animate with Authentication

echo "🔐 Launching Wan2.2 Animate with Authentication"
echo "================================================"

# Default credentials (CHANGE THESE!)
USERNAME="${WAN_USERNAME:-admin}"
PASSWORD="${WAN_PASSWORD:-wan2024}"

echo ""
echo "⚠️  SECURITY NOTICE:"
echo "   Default credentials: $USERNAME / $PASSWORD"
echo "   Change these by setting WAN_USERNAME and WAN_PASSWORD"
echo ""
echo "   Example:"
echo "   export WAN_USERNAME='myuser'"
echo "   export WAN_PASSWORD='my_secure_password'"
echo "   ./launch_with_auth.sh"
echo ""

# Activate virtual environment
source /workspace/Wan2.2/venv_wan22/bin/activate

# Change to workspace directory
cd /workspace/Wan2.2

# Launch with authentication
python app_gradio.py \
    --host 0.0.0.0 \
    --port 7860 \
    --share \
    --auth "$USERNAME:$PASSWORD" \
    --auth-message "🎬 Wan2.2 Animate - Enter your credentials"

echo ""
echo "✅ Server started with authentication!"
echo "   Access at: http://localhost:7860"
