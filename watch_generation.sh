#!/bin/bash
# Watch generation progress in real-time

echo "🔍 Watching generation progress..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor the latest process_results directory
LATEST_DIR=$(ls -td /workspace/Wan2.2/outputs/animate/process_results_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "⚠️  No active generation found"
    echo ""
    echo "💡 Tip: Start a generation from the Gradio UI first"
    exit 0
fi

echo "📁 Watching: $LATEST_DIR"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Watch for new MP4 files
while true; do
    clear
    echo "🎬 Generation Progress Monitor"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📁 Directory: $LATEST_DIR"
    echo ""
    
    # Check if video exists
    if ls "$LATEST_DIR"/*.mp4 1> /dev/null 2>&1; then
        echo "✅ Video generated!"
        ls -lh "$LATEST_DIR"/*.mp4
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        break
    else
        echo "⏳ Generating... (no MP4 file yet)"
        echo ""
        echo "📊 Directory contents:"
        ls -lh "$LATEST_DIR" 2>/dev/null || echo "   (empty)"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "💡 Refreshing in 5 seconds... (Ctrl+C to stop)"
    fi
    
    sleep 5
done

echo ""
echo "🎉 Generation complete!"
