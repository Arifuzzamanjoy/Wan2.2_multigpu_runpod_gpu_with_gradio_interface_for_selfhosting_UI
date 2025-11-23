#!/bin/bash
# Quick status check for Wan2.2 generation

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎬 Wan2.2 Generation Status Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Gradio process
echo "📡 Gradio Server:"
if pgrep -f "python app_gradio.py" > /dev/null; then
    echo "   ✅ Running (PID: $(pgrep -f 'python app_gradio.py'))"
else
    echo "   ❌ Not running"
fi
echo ""

# Check generation process
echo "🚀 Generation Process:"
if pgrep -f "torchrun.*generate.py" > /dev/null; then
    echo "   ✅ Running (PID: $(pgrep -f 'torchrun.*generate.py'))"
    echo ""
    echo "   📊 Sub-processes:"
    pgrep -f "generate.py" | while read pid; do
        echo "      - PID $pid (Rank: $(ps -p $pid -o args= | grep -oP 'rank \K\d+' || echo '?'))"
    done
else
    echo "   ❌ Not running"
fi
echo ""

# GPU status
echo "💻 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name util mem_used mem_total; do
        printf "   GPU %s: %s\n" "$idx" "$name"
        printf "      Usage: %3s%% | Memory: %s MB / %s MB\n" "$util" "$mem_used" "$mem_total"
    done
else
    echo "   ⚠️  nvidia-smi not available"
fi
echo ""

# Latest output directory
echo "📁 Latest Output:"
LATEST=$(ls -td /workspace/Wan2.2/outputs/animate/process_results_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "   Directory: $LATEST"
    if ls "$LATEST"/*.mp4 1> /dev/null 2>&1; then
        echo "   ✅ Video found:"
        ls -lh "$LATEST"/*.mp4 | awk '{printf "      %s (%s)\n", $9, $5}'
    else
        echo "   ⏳ Generating... (no MP4 yet)"
        file_count=$(ls -1 "$LATEST" 2>/dev/null | wc -l)
        echo "      Files in directory: $file_count"
    fi
else
    echo "   ⚠️  No output directory found"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Helpful commands
echo "💡 Quick Commands:"
echo "   Watch generation: ./watch_generation.sh"
echo "   Monitor GPUs:     watch -n 1 nvidia-smi"
echo "   Stop Gradio:      pkill -f 'python app_gradio.py'"
echo "   Stop generation:  pkill -f 'torchrun.*generate.py'"
echo ""
