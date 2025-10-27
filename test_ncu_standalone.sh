#!/bin/bash
set -e

echo "=== Testing NCU with Standalone Kernel ==="
echo ""

# Compile
echo "[1] Compiling simple_ncu_test.cu..."
nvcc -O3 simple_ncu_test.cu -o simple_ncu_test
echo "✓ Compilation successful"
echo ""

# Run NCU
echo "[2] Running NCU SpeedOfLight analysis..."
ncu --csv --section SpeedOfLight ./simple_ncu_test > test_output.csv
echo "✓ NCU completed"
echo ""

# Check output size
SIZE=$(wc -c < test_output.csv)
echo "[3] Output size: $SIZE bytes"
if [ $SIZE -lt 300 ]; then
    echo "✗ Output too small - NCU didn't capture data"
    echo "Content:"
    cat test_output.csv
    exit 1
else
    echo "✓ Got substantial output - NCU is working!"
    echo "First 500 characters:"
    head -c 500 test_output.csv
fi
echo ""
echo "=== NCU Test Complete ==="

