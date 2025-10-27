#!/bin/bash

# Complete NCU Profiling Script for CS 149
# Runs all NCU analyses and parses results for LLM

KERNEL_BINARY=${1:-./saxpy}
OUTPUT_DIR=${2:-./ncu_output}
KERNEL_SOURCE=${3:-saxpy.cu}

if [ ! -f "$KERNEL_BINARY" ]; then
    echo "Error: Kernel binary '$KERNEL_BINARY' not found"
    echo "Usage: $0 <kernel_binary> [output_dir] [kernel_source]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "NCU Profiling for CS 149"
echo "Kernel: $KERNEL_BINARY"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Full report (binary format)
echo ""
echo "Step 1: Generating full NCU report..."
/usr/local/cuda/bin/ncu --set full --export "$OUTPUT_DIR/full_report" "$KERNEL_BINARY"

# Speed of Light (high-level bottleneck)
echo ""
echo "Step 2: Speed of Light analysis..."
/usr/local/cuda/bin/ncu --csv --section SpeedOfLight "$KERNEL_BINARY" > "$OUTPUT_DIR/speed_of_light.csv"

# Memory analysis
echo ""
echo "Step 3: Memory workload analysis..."
/usr/local/cuda/bin/ncu --csv --section MemoryWorkloadAnalysis "$KERNEL_BINARY" > "$OUTPUT_DIR/memory_analysis.csv"

# Compute analysis
echo ""
echo "Step 4: Compute workload analysis..."
/usr/local/cuda/bin/ncu --csv --section ComputeWorkloadAnalysis "$KERNEL_BINARY" > "$OUTPUT_DIR/compute_analysis.csv"

# Occupancy analysis
echo ""
echo "Step 5: Occupancy analysis..."
/usr/local/cuda/bin/ncu --csv --section Occupancy "$KERNEL_BINARY" > "$OUTPUT_DIR/occupancy.csv"

# Scheduler statistics (stall reasons)
echo ""
echo "Step 6: Scheduler statistics..."
/usr/local/cuda/bin/ncu --csv --section SchedulerStats "$KERNEL_BINARY" > "$OUTPUT_DIR/scheduler_stats.csv"

# Warp state statistics (execution efficiency)
echo ""
echo "Step 7: Warp state statistics..."
/usr/local/cuda/bin/ncu --csv --section WarpStateStats "$KERNEL_BINARY" > "$OUTPUT_DIR/warp_stats.csv"

echo ""
echo "=========================================="
echo "NCU Profiling Complete!"
echo "=========================================="

# Parse results if parser exists
if [ -f "parse_ncu.py" ]; then
    echo ""
    echo "Step 8: Parsing results for LLM..."
    python3 parse_ncu.py "$OUTPUT_DIR" "$KERNEL_SOURCE"
else
    echo ""
    echo "parse_ncu.py not found. Skipping parsing step."
fi

echo ""
echo "=========================================="
echo "All Done!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/full_report.ncu-rep"
echo "  - $OUTPUT_DIR/*.csv"
echo "  - $OUTPUT_DIR/parsed_metrics.json"
echo "  - $OUTPUT_DIR/llm_prompt.txt"
echo ""

