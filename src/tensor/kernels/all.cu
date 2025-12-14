#include "common.cuh"

extern "C" __global__ void allContiguous(
    const bool *a,
    int *result,
    int n
) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int val = (i < n) ? (int)a[i] : 1;

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val &= __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ int warpResults[32];  // Max 32 warps per block
    int warpId = tid / warpSize;
    int lane = tid % warpSize;

    if (lane == 0) {
        warpResults[warpId] = val;
    }
    __syncthreads();

    // First warp reduces all warp results
    if (warpId == 0) {
        val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpResults[tid] : 1;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val &= __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (tid == 0) {
            atomicAnd(result, val);
        }
    }
}

extern "C" __global__ void allStrided(
    const bool *a,
    TensorInfo a_info,
    int *result,
    KernelArgs args
) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int val = 1;
    if (i < args.elems) {
      int a_off = stridedOffset(i, args.shape, a_info.strides, args.ndim);
      val = (int)a[a_off];
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val &= __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ int warpResults[32];  // Max 32 warps per block
    int warpId = tid / warpSize;
    int lane = tid % warpSize;

    if (lane == 0) {
        warpResults[warpId] = val;
    }
    __syncthreads();

    // First warp reduces all warp results
    if (warpId == 0) {
        val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpResults[tid] : 1;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val &= __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (tid == 0) {
            atomicAnd(result, val);
        }
    }
}
