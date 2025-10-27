# NCU Profiling System - Integration Guide

## Concept Overview

This system automates CUDA kernel performance analysis using NVIDIA Nsight Compute (NCU). Here's how it works:

```
Student submits CUDA kernel (.cu file)
          ↓
    [compile.sh] compiles to executable
          ↓
    [run_ncu_complete.sh] profiles with NCU → generates 6 CSV files
          ↓
    [parse_ncu.py] extracts metrics & identifies bottlenecks
          ↓
    Outputs: parsed_metrics.json + llm_prompt.txt
          ↓
    Feed to LLM (GPT-4/Claude) for optimization suggestions
```

### Why This is Useful

- **Automated Performance Analysis**: No manual NCU profiling needed
- **Structured Output**: Metrics in JSON format for easy processing
- **LLM-Ready Prompts**: Pre-formatted for AI optimization suggestions
- **GitHub Actions Integration**: Automatic feedback on student submissions
- **Production-Grade**: Handles edge cases, supports all NCU metric sections

---

## Files to Add to Your Repository

### 1. `compile.sh` - CUDA Compilation Script

```bash
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
```

**Make executable:**
```bash
chmod +x compile.sh
```

**Usage:**
```bash
./compile.sh               # Compiles saxpy.cu
./compile.sh my_kernel.cu  # Compiles custom kernel
```

---

### 2. `run_ncu_complete.sh` - NCU Profiling Script

```bash
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
```

**Make executable:**
```bash
chmod +x run_ncu_complete.sh
```

**Usage:**
```bash
./run_ncu_complete.sh                          # Profile ./saxpy
./run_ncu_complete.sh ./my_kernel              # Profile custom binary
./run_ncu_complete.sh ./my_kernel output_dir   # Custom output dir
```

---

### 3. `parse_ncu.py` - Metric Parser

This is a large Python file. See the complete code in the next section.

**Key Features:**
- Parses 6 NCU CSV output sections
- Identifies bottlenecks automatically (memory, compute, occupancy, divergence)
- Generates structured JSON output
- Creates LLM-ready prompts with kernel code
- Handles NCU header lines and edge cases

**Usage:**
```bash
python3 parse_ncu.py ncu_output saxpy.cu
```

**Outputs:**
- `ncu_output/parsed_metrics.json` - Structured performance data
- `ncu_output/llm_prompt.txt` - Ready for GPT-4/Claude

---

### 4. Example CUDA Kernel (`saxpy.cu`)

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(int n, float a, float *x, float *y) {
    float *d_x, *d_y;
    
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);
    
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(void) {
    int N = 1 << 20; // 1M elements
    float a = 2.0f;
    
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    saxpy(N, a, x, y);
    
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 4.0f));
    }
    printf("Max error: %f\n", maxError);
    
    free(x);
    free(y);
    
    return 0;
}
```

---

## Complete `parse_ncu.py` Code

See attached file or copy from: `/home/ubuntu/cs149-ncu/parse_ncu.py`

Key sections:
- **NCUParser class**: Main parser logic
- **parse_csv()**: Handles NCU CSV format with header skipping
- **extract_metric_value()**: Extracts specific metrics by name
- **6 parse methods**: One for each NCU section
- **generate_recommendations()**: Auto-generates optimization tips
- **to_llm_prompt()**: Creates formatted LLM input

---

## GitHub Actions Integration

### Setup Self-Hosted Runner

**Why?** GitHub's hosted runners don't have GPUs. You need a self-hosted runner on a GPU instance.

**Requirements:**
- AWS g6.xlarge (NVIDIA L4 GPU) or similar
- Ubuntu 22.04
- CUDA Toolkit installed
- NCU installed

**Setup Steps:**

```bash
# 1. Download GitHub Actions runner
cd /path/to/your/workspace
curl -o actions-runner-linux-x64-2.329.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.329.0/actions-runner-linux-x64-2.329.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.329.0.tar.gz

# 2. Get registration token from GitHub
# Go to: Settings → Actions → Runners → New self-hosted runner
# Copy the token

# 3. Configure runner
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN --replace

# 4. Start runner
./run.sh

# For production, use systemd:
sudo ./svc.sh install
sudo ./svc.sh start
```

### GitHub Workflow File

Create `.github/workflows/ncu-profile.yml`:

```yaml
name: CUDA Kernel Profiling

on:
  push:
    paths:
      - '**.cu'
  pull_request:
    paths:
      - '**.cu'
  workflow_dispatch:

jobs:
  profile:
    runs-on: self-hosted
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: List CUDA files
        id: find_kernels
        run: |
          KERNELS=$(find . -name "*.cu" -type f | tr '\n' ' ')
          echo "kernels=$KERNELS" >> $GITHUB_OUTPUT
      
      - name: Compile kernels
        run: |
          for kernel in ${{ steps.find_kernels.outputs.kernels }}; do
            echo "Compiling $kernel"
            ./compile.sh "$kernel"
          done
      
      - name: Profile with NCU
        run: |
          for kernel in ${{ steps.find_kernels.outputs.kernels }}; do
            binary=$(basename "$kernel" .cu)
            echo "Profiling $binary"
            ./run_ncu_complete.sh "./$binary" "ncu_output_$binary" "$kernel"
          done
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: profiling-results
          path: |
            ncu_output_*/parsed_metrics.json
            ncu_output_*/llm_prompt.txt
      
      - name: Comment on PR (optional)
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const path = require('path');
            
            // Find all LLM prompt files
            const promptFiles = fs.readdirSync('.')
              .filter(f => f.startsWith('ncu_output_'))
              .map(dir => path.join(dir, 'llm_prompt.txt'))
              .filter(f => fs.existsSync(f));
            
            let comment = '## 🚀 NCU Profiling Results\n\n';
            
            promptFiles.forEach(file => {
              const content = fs.readFileSync(file, 'utf8');
              const dir = path.dirname(file);
              comment += `### ${dir}\n\n\`\`\`\n${content}\n\`\`\`\n\n`;
            });
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

**What this workflow does:**
1. Triggers on `.cu` file changes
2. Finds all CUDA kernels
3. Compiles each one
4. Profiles with NCU
5. Uploads results as artifacts
6. (Optional) Comments results on PRs

---

## Quick Integration Checklist

### For Your Friend to Do:

**1. Copy Files to Repository**
```bash
cd /path/to/their/repo

# Copy profiling scripts
cp /path/to/cs149-ncu/compile.sh .
cp /path/to/cs149-ncu/run_ncu_complete.sh .
cp /path/to/cs149-ncu/parse_ncu.py .

# Make scripts executable
chmod +x compile.sh run_ncu_complete.sh
```

**2. Create GitHub Workflow**
```bash
mkdir -p .github/workflows
# Copy the workflow YAML from above into:
# .github/workflows/ncu-profile.yml
```

**3. Set Up GPU Instance**
- Launch AWS g6.xlarge with CUDA Deep Learning AMI
- Install NCU (comes with CUDA Toolkit)
- Enable GPU profiling:
  ```bash
  sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'
  sudo update-initramfs -u
  sudo reboot
  ```

**4. Configure Self-Hosted Runner**
```bash
# On the GPU instance:
cd /home/ubuntu
mkdir actions-runner && cd actions-runner

# Download runner
curl -o actions-runner-linux-x64-2.329.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.329.0/actions-runner-linux-x64-2.329.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.329.0.tar.gz

# Get token from GitHub: Settings → Actions → Runners → New self-hosted runner
./config.sh --url https://github.com/OWNER/REPO --token TOKEN --replace

# Start runner (use systemd for production)
sudo ./svc.sh install
sudo ./svc.sh start
```

**5. Test the System**
```bash
# Local test first:
./compile.sh saxpy.cu
./run_ncu_complete.sh ./saxpy
cat ncu_output/llm_prompt.txt

# Then push a .cu file to trigger GitHub Actions
git add test_kernel.cu
git commit -m "Test NCU profiling"
git push
```

---

## Output Format Examples

### `parsed_metrics.json`
```json
{
  "metadata": {},
  "summary": {
    "SM %": 16.4,
    "Memory %": 86.3,
    "DRAM %": 86.3,
    "L1/TEX Cache %": 41.8,
    "L2 Cache %": 26.9
  },
  "bottlenecks": [
    {
      "type": "Memory %",
      "utilization": 86.3,
      "severity": "high"
    }
  ],
  "metrics": {
    "compute": { "SM Busy": 15.8, "Issue Slots Busy": 11.2 },
    "memory": { "Mem Busy": 49.9, "L1/TEX Hit Rate": 32.9 },
    "occupancy": { "Achieved Occupancy": 85.4 },
    "control_flow": { "No Eligible": 91.6 }
  },
  "recommendations": [...]
}
```

### `llm_prompt.txt`
```
# CUDA Kernel Optimization Analysis

## Kernel Code
[kernel source code]

## Performance Summary
- SM %: 16.4%
- Memory %: 86.3%
- DRAM %: 86.3%

## Identified Bottlenecks
1. **Memory %** (Severity: high)
   - utilization: 86.3

## Detailed Metrics
[detailed metrics]

## Recommended Optimizations
[specific suggestions]
```

---

## Metrics Captured

The system profiles 6 key performance areas:

1. **Speed of Light** - High-level bottleneck identification
2. **Memory Analysis** - Cache hit rates, bandwidth usage
3. **Compute Analysis** - SM utilization, IPC
4. **Occupancy** - Warp occupancy (achieved vs theoretical)
5. **Scheduler Stats** - Stall reasons, eligible warps
6. **Warp Stats** - Thread divergence, execution efficiency

---

## Troubleshooting

### "Permission denied" from NCU
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'
sudo update-initramfs -u
sudo reboot
```

### Empty parsed_metrics.json
- Check CSV files exist: `ls -lh ncu_output/*.csv`
- Verify NCU ran: `cat ncu_output/speed_of_light.csv | head -20`

### Runner not receiving jobs
- Check runner status: GitHub → Settings → Actions → Runners
- Verify it's "Idle" (green)
- Check labels match workflow: `runs-on: self-hosted`

### Compilation fails
- Check CUDA arch: `nvidia-smi` → adjust `-arch=sm_XX` in compile.sh
- Verify nvcc: `nvcc --version`

---

## Cost Estimate

**AWS g6.xlarge** (NVIDIA L4):
- On-demand: ~$0.84/hour
- Spot: ~$0.30/hour (70% savings)
- Per student submission: < $0.01 (10-30 seconds)

**Recommendation**: Use spot instances with auto-scaling

---

## Summary for Your Friend

**What you're getting:**
- Automated CUDA kernel performance analysis
- Structured metrics in JSON
- LLM-ready optimization prompts
- GitHub Actions integration for automated feedback

**What they need:**
- GPU instance (AWS g6.xlarge or similar)
- CUDA Toolkit + NCU
- GitHub repository with Actions enabled
- 30 minutes for setup

**Files to copy:**
- `compile.sh`
- `run_ncu_complete.sh`
- `parse_ncu.py`
- `.github/workflows/ncu-profile.yml`

**Expected result:**
Students push `.cu` files → automatic profiling → performance feedback in ~30 seconds

---

For questions or issues, refer to the main README.md in the cs149-ncu repository.

