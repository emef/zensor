#ifndef KERNELS_H
#define KERNELS_H

#define MAX_DIMS 8

struct TensorInfo {
    int strides[MAX_DIMS];
    int ndim;
};

struct KernelArgs {
    int shape[MAX_DIMS];
    int ndim;
    int elems;
};

__device__ __forceinline__ int stridedOffset(
    int linear_idx,
    const int *shape,
    const int *strides,
    int ndim
  ) {
    int offset = 0;
    #pragma unroll
    for (int d = MAX_DIMS - 1; d >= 0; d--) {
        if (d < ndim) {
            offset += (linear_idx % shape[d]) * strides[d];
            linear_idx /= shape[d];
        }
    }
    return offset;
}

#endif

