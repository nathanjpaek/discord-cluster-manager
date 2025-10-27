//!POPCORN leaderboard identity_cuda

#include <array>
#include <vector>
#include "task.h"
#include "utils.h"

__global__ void copy_kernel(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = input[idx];
    }
}

output_t custom_kernel(input_t data)
{
    // Allocate GPU memory
    int N = data.size();
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Copy input to GPU
    cudaMemcpy(d_input, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel - THIS IS WHERE YOUR KERNEL RUNS ON GPU!
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    copy_kernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    
    // Copy result back to CPU
    output_t result(N);
    cudaMemcpy(result.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}
