#include <stdio.h>

__global__ void simple_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}

int main() {
    int N = 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    simple_kernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    printf("Done\n");
    return 0;
}
