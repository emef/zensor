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

#define KERNEL_ELEMENTWISE_STRIDED(name, impl, suffix, type) \
    extern "C" __global__ void name##_##suffix( \
        const type *a, TensorInfo a_info, \
        const type *b, TensorInfo b_info, \
        type *out, \
        KernelArgs args \
    ) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i < args.elems) impl<type>(a, a_info, b, b_info, out, args, i); \
    }

#define KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, suffix, type) \
    extern "C" __global__ void name##_##suffix( \
        const type *a, \
        const type *b, \
        type *out, \
        int n \
    ) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i < n) impl<type>(a, b, out, i); \
    }


#define FOR_ALL_TYPES_ELEMENTWISE_CONTIGUOUS(name, impl) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, bool, bool) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, f32, float) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, f64, double) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, i8, int8_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, i16, int16_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, i32, int32_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, i64, int64_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, u8, uint8_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, u16, uint16_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, u32, uint32_t) \
    KERNEL_ELEMENTWISE_CONTIGUOUS(name, impl, u64, uint64_t)

#define FOR_ALL_TYPES_ELEMENTWISE_STRIDED(name, impl) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, bool, bool) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, f32, float) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, f64, double) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, i8, int8_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, i16, int16_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, i32, int32_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, i64, int64_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, u8, uint8_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, u16, uint16_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, u32, uint32_t) \
    KERNEL_ELEMENTWISE_STRIDED(name, impl, u64, uint64_t)

#endif


