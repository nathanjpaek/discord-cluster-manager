//!POPCORN leaderboard identity_cuda


// NOTE OUTPUT IS HARDCODED AT 16KB

#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include "task.h"
#include "utils.h"

// ============================================================================
// YOUR KERNEL IMPLEMENTATION HERE
// ============================================================================

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    // Simple naive implementation - replace with your optimized version!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// REFERENCE IMPLEMENTATION (for correctness checking)
// ============================================================================

void reference_matmul_cpu(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// CORRECTNESS CHECKING
// ============================================================================

struct CorrectnessResult {
    bool passed;
    float max_error;
    float avg_error;
    int total_elements;
    
    void print() const {
        if (passed) {
            printf("✓ CORRECTNESS: PASS (max_err=%.6e, avg_err=%.6e)\n", 
                   max_error, avg_error);
        } else {
            printf("✗ CORRECTNESS: FAIL (max_err=%.6e, avg_err=%.6e)\n", 
                   max_error, avg_error);
        }
    }
};

CorrectnessResult check_correctness(const float* result, const float* reference, 
                                     int size, float tolerance = 1e-4f) {
    CorrectnessResult res;
    res.max_error = 0.0f;
    res.avg_error = 0.0f;
    res.total_elements = size;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(result[i] - reference[i]);
        res.max_error = fmax(res.max_error, diff);
        res.avg_error += diff;
    }
    res.avg_error /= size;
    res.passed = (res.max_error < tolerance);
    
    return res;
}

// ============================================================================
// PERFORMANCE MEASUREMENT
// ============================================================================

struct PerformanceResult {
    float time_ms;
    float gflops;
    float bandwidth_gb_s;
    
    void print() const {
        printf("⚡ PERFORMANCE: %.3f ms | %.2f GFLOPS | %.2f GB/s\n",
               time_ms, gflops, bandwidth_gb_s);
    }
};

PerformanceResult measure_performance(
    void (*kernel_launch)(const float*, const float*, float*, int, int, int),
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, int warmup_iters = 3, int timing_iters = 10) {
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel_launch(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < timing_iters; i++) {
        kernel_launch(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / timing_iters;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Calculate metrics
    PerformanceResult res;
    res.time_ms = avg_ms;
    
    // FLOPS: 2*M*N*K operations (multiply-add)
    long long flops = 2LL * M * N * K;
    res.gflops = (flops / (avg_ms / 1000.0f)) / 1e9f;
    
    // Bandwidth: reading A (M*K) + B (K*N) + writing C (M*N)
    long long bytes = sizeof(float) * (M * K + K * N + M * N);
    res.bandwidth_gb_s = (bytes / (avg_ms / 1000.0f)) / 1e9f;
    
    return res;
}

// ============================================================================
// KERNEL LAUNCHER (wraps your kernel with launch config)
// ============================================================================

void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K) {
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ============================================================================
// MAIN TESTING FUNCTION
// ============================================================================

output_t custom_kernel(input_t data) {
    // Extract test size from input - use sqrt to get reasonable matrix dimensions
    int size = data.size();
    
    // For matmul: if input size is 128, use 11x11 matrices (fits in memory)
    // Adjust this based on your kernel's memory requirements
    int dim = (int)sqrt((float)size);
    if (dim < 8) dim = 8;  // Minimum size
    if (dim > 64) dim = 64;  // Maximum size to avoid OOM
    
    int M = dim;
    int N = dim;
    int K = dim;
    
    printf("\n=== KERNEL TEST: MatMul %dx%dx%d ===\n\n", M, N, K);
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Time reference implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    reference_matmul_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Run your kernel
    launch_matmul(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check correctness
    CorrectnessResult correctness = check_correctness(
        h_C.data(), h_C_ref.data(), M * N, 1e-3f
    );
    correctness.print();
    
    float speedup = 0.0f;
    if (correctness.passed) {
        PerformanceResult perf = measure_performance(
            launch_matmul, d_A, d_B, d_C, M, N, K
        );
        
        speedup = cpu_time_ms / perf.time_ms;
        long long total_bytes = sizeof(float) * (M * K + K * N + M * N);
        float memory_mb = total_bytes / (1024.0f * 1024.0f);
        
        printf("⚡ PERFORMANCE: %.3f ms | %.2f GFLOPS | %.2f GB/s\n",
               perf.time_ms, perf.gflops, perf.bandwidth_gb_s);
        printf("🚀 SPEEDUP: %.1fx (CPU: %.1fms, GPU: %.1fms)\n", 
               speedup, cpu_time_ms, perf.time_ms);
        printf("📊 Memory: %.2f MB | Bandwidth: %.2f GB/s\n", memory_mb, perf.bandwidth_gb_s);
        
        float peak_gflops = 242.0f;
        float compute_eff = (perf.gflops / peak_gflops) * 100.0f;
        printf("📈 Efficiency: %.1f%% compute | Arithmetic Intensity: %.1f FLOPS/byte\n",
               compute_eff, (2.0f * M * N * K) / (float)total_bytes);
    } else {
        printf("⚠️ FAILED - Skipping performance test\n");
    }
    
    printf("\n");
    if (correctness.passed) {
        printf("✅ PASSED | Speedup: %.1fx | Memory: %.2fMB\n", 
               speedup, (sizeof(float) * (M * K + K * N + M * N)) / (1024.0f * 1024.0f));
    } else {
        printf("❌ FAILED | Max Error: %.6e\n", correctness.max_error);
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Return the result (for compatibility with test harness)
    output_t result(M * N);
    for (int i = 0; i < M * N; i++) {
        result[i] = h_C[i];
    }
    
    return result;
}

