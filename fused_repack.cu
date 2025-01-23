#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_repack_merge_kernel(
    const float* __restrict__ input,
    const bool* __restrict__ mask,
    float* __restrict__ output,
    bool* __restrict__ output_mask,
    const int B,
    const int N,
    const int M,
    const int d,
    int* __restrict__ current_indices,
    const int* __restrict__ pruned_counts
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) unsigned char smem[];
    float* sum = reinterpret_cast<float*>(smem);

    // Initialize shared memory
    for (int i = tid; i < d; i += blockDim.x) sum[i] = 0.0f;
    __syncthreads();

    // Phase 1: Accumulate pruned tokens
    for (int n = tid; n < N; n += blockDim.x) {
        if (!mask[b * N + n]) {
            for (int i = 0; i < d; ++i) {
                atomicAdd(&sum[i], input[b * N * d + n * d + i]);
            }
        }
    }
    __syncthreads();

    // Phase 2: Write merged token (if pruned tokens exist)
    if (pruned_counts[b] > 0 && tid == 0) {
        const int merged_pos = N - pruned_counts[b];  // = kept_counts[b]
        for (int i = 0; i < d; ++i) {
            output[b * M * d + merged_pos * d + i] = sum[i] / pruned_counts[b];
        }
        output_mask[b * M + merged_pos] = true;
    }
    __syncthreads();

    // Phase 3: Copy kept tokens using thread-safe indices
    __shared__ int shared_idx;
    if (tid == 0) {
        shared_idx = current_indices[b];  // Starts at 0 per batch
    }
    __syncthreads();

    for (int n = tid; n < N; n += blockDim.x) {
        if (mask[b * N + n]) {
            const int output_pos = atomicAdd(&shared_idx, 1);
            for (int i = 0; i < d; ++i) {
                output[b * M * d + output_pos * d + i] = input[b * N * d + n * d + i];
            }
            output_mask[b * M + output_pos] = true;
        }
    }

    // Update current_indices for future kernels
    if (tid == 0) {
        current_indices[b] = shared_idx;
    }
}

void fused_repack_merge_impl(
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor output,
    torch::Tensor output_mask,
    torch::Tensor current_indices,
    torch::Tensor pruned_counts,
    int M
) {
    const int B = input.size(0);
    const int N = input.size(1);
    const int d = input.size(2);

    const dim3 blocks(B);
    const dim3 threads(256);
    const size_t shared_mem = d * sizeof(float);

    fused_repack_merge_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        output_mask.data_ptr<bool>(),
        B, N, M, d,
        current_indices.data_ptr<int>(),
        pruned_counts.data_ptr<int>()
    );
}