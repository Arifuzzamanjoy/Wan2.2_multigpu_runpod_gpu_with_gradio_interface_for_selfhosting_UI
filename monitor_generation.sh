#!/bin/bash
# Monitor Wan2.2-Animate generation progress

echo "=== Wan2.2-Animate Generation Monitor ==="
echo ""

# Check if generation is running
if pgrep -f "generate.py.*animate-14B" > /dev/null; then
    PID=$(pgrep -f "generate.py.*animate-14B")
    echo "✓ Generation is RUNNING (PID: $PID)"
    
    # Get runtime
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "⏱  Runtime: $RUNTIME"
    
    # Get memory usage
    MEM=$(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "💾 Memory: $MEM"
    
    # Check GPU usage
    echo ""
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "   GPU %s (%s): %s%% util, %s/%s MB\n", $1, $2, $3, $4, $5}'
    
    # Check for output files
    echo ""
    echo "📁 Output directory: /workspace/Wan2.2/outputs/animate/"
    if [ -d "/workspace/Wan2.2/outputs/animate/" ]; then
        FILES=$(find /workspace/Wan2.2/outputs/animate/ -type f -name "*.mp4" -o -name "*.gif" 2>/dev/null | wc -l)
        if [ $FILES -gt 0 ]; then
            echo "   ✓ Found $FILES output file(s)!"
            find /workspace/Wan2.2/outputs/animate/ -type f \( -name "*.mp4" -o -name "*.gif" \) -exec ls -lh {} \;
        else
            echo "   ⏳ Generating... (no output files yet)"
        fi
    fi
else
    echo "❌ Generation is NOT running"
    echo ""
    echo "Recent output files:"
    find /workspace/Wan2.2/outputs/animate/ -type f \( -name "*.mp4" -o -name "*.gif" \) -mmin -60 -exec ls -lht {} \; 2>/dev/null | head -5
fi

echo ""
echo "Run: watch -n 5 /workspace/Wan2.2/monitor_generation.sh"
echo "To stop monitoring: Ctrl+C"
