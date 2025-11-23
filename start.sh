#!/bin/bash
# QUICK START - Wan2.2 Animate Gradio Interface
# Just run this script and go to http://localhost:7860

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  🎬 Wan2.2 Animate - Gradio Interface"
echo "═══════════════════════════════════════════════════════"
echo ""

# Activate environment
source /workspace/Wan2.2/venv_wan22/bin/activate

# Go to workspace
cd /workspace/Wan2.2

echo "🚀 Starting Gradio server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Access the interface at:"
echo "  👉 http://localhost:7860"
echo ""
echo "  Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Launch
python app_gradio.py --host 0.0.0.0 --port 7860

# Note: Remove --share flag for local-only access
# Add --auth "user:pass" for password protection
