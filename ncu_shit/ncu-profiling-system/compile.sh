#!/bin/bash

# Compilation script for CUDA kernels

KERNEL_FILE=${1:-saxpy.cu}
OUTPUT_NAME=$(basename "$KERNEL_FILE" .cu)

echo "Compiling $KERNEL_FILE..."

nvcc -O3 -arch=sm_89 -o "$OUTPUT_NAME" "$KERNEL_FILE"

if [ $? -eq 0 ]; then
    echo "Compilation successful: ./$OUTPUT_NAME"
else
    echo "Compilation failed"
    exit 1
fi

