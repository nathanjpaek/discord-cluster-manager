#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

int main() {
    int N = 1048576; // 1M elements for measurable work
    float *d_data;
    
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Launch kernel multiple times to ensure NCU captures it
    for (int i = 0; i < 10; i++) {
        test_kernel<<<1024, 256>>>(d_data, N);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_data);
    
    printf("Kernel executed successfully\n");
    return 0;
}

