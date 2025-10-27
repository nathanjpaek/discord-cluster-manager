# NCU Profiling Guide

This system integrates NVIDIA Nsight Compute (NCU) profiling into the CUDA kernel submission workflow, allowing you to get detailed performance analysis and optimization recommendations for your GPU kernels.

## What is NCU Profiling?

NCU (NVIDIA Nsight Compute) is a command-line profiler that provides detailed performance metrics for CUDA kernels, including:
- Memory throughput and bottlenecks
- Compute utilization
- Occupancy analysis
- Warp execution efficiency
- Scheduler statistics

## Quick Start

### Basic Usage

```bash
python scripts/submit_github.py <kernel.cu> --mode profile --run-id "profile-$(date +%s)"
```

### Example

```bash
python scripts/submit_github.py test_kernel.cu --mode profile --run-id "saxpy-profile-$(date +%s)"
```

## Kernel Requirements

Your CUDA kernel must be **standalone** with a `main()` function. Example:

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void my_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *h_data = (float*)malloc(N * sizeof(float));
    float *d_data;
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    // Allocate device memory
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    my_kernel<<<numBlocks, blockSize>>>(d_data, N);
    
    // Copy results back
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    printf("Kernel completed successfully\n");
    return 0;
}
```

## Output Files

After profiling completes, two files are saved locally:

### 1. `ncu_llm_prompt.txt`
Complete LLM-ready optimization analysis including:
- Your kernel source code
- Performance summary (SM %, Memory %, etc.)
- Identified bottlenecks with severity levels
- Detailed metrics by category (compute, memory, occupancy, control flow)
- Actionable optimization recommendations

**Use this file:** Copy the content and paste it into Claude/ChatGPT for optimization suggestions!

### 2. `ncu_parsed_metrics.json`
Structured JSON with all parsed NCU metrics:
```json
{
  "metadata": {},
  "summary": {
    "SM %": 45.2,
    "Memory %": 78.5,
    "DRAM %": 65.3,
    ...
  },
  "bottlenecks": [
    {
      "type": "Memory",
      "utilization": 78.5,
      "severity": "high"
    }
  ],
  "metrics": {
    "compute": {...},
    "memory": {...},
    "occupancy": {...},
    "control_flow": {...}
  },
  "recommendations": [...]
}
```

## Profiling Sections

The system profiles 6 key NCU metric sections:

1. **SpeedOfLight** - High-level bottleneck identification
2. **MemoryWorkloadAnalysis** - Memory bandwidth and cache usage
3. **ComputeWorkloadAnalysis** - SM utilization and instruction throughput
4. **Occupancy** - Active warps and occupancy limiting factors
5. **SchedulerStats** - Warp scheduling efficiency
6. **WarpStateStats** - Warp divergence and execution efficiency

## Example Output

```
=== NCU Profiling Results ===

┌─ Performance Summary ──────────────────────────────────────────────────┐
│ SM %:                    45.23%
│ Memory %:                78.56%
│ DRAM %:                  65.12%
│ L1/TEX Cache %:          42.89%
│ L2 Cache %:              58.34%
└────────────────────────────────────────────────────────────────────────┘

┌─ Identified Bottlenecks ───────────────────────────────────────────────┐
│ 1. Memory % (Severity: high)
│    Utilization: 78.56%
└────────────────────────────────────────────────────────────────────────┘

┌─ Recommended Optimizations ────────────────────────────────────────────┐
│ 1. Memory Optimization
│    Issue: High memory throughput utilization
│    Suggestion: Use shared memory to reduce global memory accesses
│ 
│ 2. Cache Optimization
│    Issue: Suboptimal L1/TEX cache utilization
│    Suggestion: Coalesce memory accesses for better bandwidth utilization
└────────────────────────────────────────────────────────────────────────┘

┌─ LLM Prompt & Metrics Saved ───────────────────────────────────────────┐
│ NCU profiling results saved locally:
│   • ncu_llm_prompt.txt - Full LLM optimization analysis
│   • ncu_parsed_metrics.json - Parsed NCU metrics (JSON)
└────────────────────────────────────────────────────────────────────────┘
```

## Modes Comparison

| Mode | Purpose | Output |
|------|---------|--------|
| `test` (default) | Run correctness tests | Pass/fail results |
| `benchmark` | Measure performance | Timing statistics |
| `profile` | Detailed profiling | NCU metrics + LLM prompt |

## Advanced Usage

### Custom Run ID
```bash
python scripts/submit_github.py kernel.cu --mode profile --run-id "my-experiment-1"
```

### View Results
```bash
# View LLM prompt
cat ncu_llm_prompt.txt

# View parsed metrics
cat ncu_parsed_metrics.json | jq
```

### Integration with LLMs
```bash
# Copy prompt to clipboard (macOS)
cat ncu_llm_prompt.txt | pbcopy

# Then paste into Claude/ChatGPT for optimization advice
```

## Troubleshooting

### "No NCU profiling data found in artifacts"
- Ensure your kernel has a `main()` function
- Check GitHub Actions logs for compilation errors
- Verify the kernel actually launches CUDA kernels

### "Compilation failed: multiple definition of main"
- Make sure you're using `--mode profile` (not `test` or `benchmark`)
- Profile mode automatically handles standalone kernels

### Files not created
- The files are only created **after** the GitHub Actions job completes
- Wait for the "conclusion: success" message
- Files are saved in your current working directory

## Implementation Details

### Workflow
1. Detects standalone kernel (has `main()` function)
2. Skips test harness (`eval.cu`) compilation for profile mode
3. Compiles kernel as standalone executable
4. Runs NCU profiling with 6 metric sections
5. Parses CSV outputs into structured JSON
6. Generates LLM-ready optimization prompt
7. Uploads artifacts to GitHub Actions
8. Downloads and saves results locally

### NCU Command
```bash
ncu --csv --section <SectionName> ./submission_standalone
```

Sections profiled: SpeedOfLight, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, Occupancy, SchedulerStats, WarpStateStats

## Tips

1. **Start with SpeedOfLight metrics** - Identifies the primary bottleneck (compute vs memory bound)
2. **Focus on high-severity bottlenecks first** - Address the biggest issues before micro-optimizations
3. **Use the LLM prompt** - It provides context-aware recommendations based on your specific kernel
4. **Compare before/after** - Profile after each optimization to measure improvement
5. **Watch occupancy** - Low occupancy often indicates resource limitations (registers, shared memory)

## See Also

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- `test_kernel.cu` - Example standalone SAXPY kernel with profiling

