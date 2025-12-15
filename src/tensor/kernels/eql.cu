#include <cstdint>
#include <assert.h>

#include "common.cuh"


template <typename T>
__device__ __forceinline__ void eqlStridedImpl(
    const T *a, TensorInfo a_info,
    const T *b, TensorInfo b_info,
    bool *out,
    KernelArgs args, int i
) {
    int a_off = stridedOffset(i, args.shape, a_info.strides, args.ndim);
    int b_off = stridedOffset(i, args.shape, b_info.strides, args.ndim);
    out[i] = a[a_off] == b[b_off];
}

#define KERNEL_BOOL_OUT(name, impl, suffix, type) \
    extern "C" __global__ void name##_##suffix( \
        const type *a, TensorInfo a_info, \
        const type *b, TensorInfo b_info, \
        bool *out, \
        KernelArgs args \
    ) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i < args.elems) impl<type>(a, a_info, b, b_info, out, args, i); \
    }

#define FOR_ALL_TYPES_BOOL_OUT(name, impl) \
    KERNEL_BOOL_OUT(name, impl, bool, bool) \
    KERNEL_BOOL_OUT(name, impl, f32, float) \
    KERNEL_BOOL_OUT(name, impl, f64, double) \
    KERNEL_BOOL_OUT(name, impl, i8, int8_t) \
    KERNEL_BOOL_OUT(name, impl, i16, int16_t) \
    KERNEL_BOOL_OUT(name, impl, i32, int32_t) \
    KERNEL_BOOL_OUT(name, impl, i64, int64_t) \
    KERNEL_BOOL_OUT(name, impl, u8, uint8_t) \
    KERNEL_BOOL_OUT(name, impl, u16, uint16_t) \
    KERNEL_BOOL_OUT(name, impl, u32, uint32_t) \
    KERNEL_BOOL_OUT(name, impl, u64, uint64_t)

FOR_ALL_TYPES_BOOL_OUT(eqlStrided, eqlStridedImpl)


