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
    return data;
}

// curl -X POST \
  -H "X-Popcorn-Cli-Id: test-user-123" \
  -F "file=@/Users/willychan/Desktop/projects/discord-cluster-manager/test_kernel.cu" \
  "http://184.72.131.76:8000/identity_cuda/NVIDIA/test"


  // # Replace YOUR_AWS_IP with your actual IP
// curl http://184.72.131.76:8000/leaderboards