#include <cstdint>
#include "common.cuh"

template <typename T>
 __device__ __forceinline__ void addContiguousImpl(
    const T *a,
    const T *b,
    T *out,
    int i
) {
    out[i] = a[i] + b[i];
}

template <typename T>
__device__ void addStridedImpl(
    const T *a, TensorInfo a_info,
    const T *b, TensorInfo b_info,
    T *out,
    KernelArgs args,
    int i
) {
    int a_off = stridedOffset(i, args.shape, a_info.strides, args.ndim);
    int b_off = stridedOffset(i, args.shape, b_info.strides, args.ndim);
    out[i] = a[a_off] + b[b_off];
}

FOR_ALL_TYPES_ELEMENTWISE_CONTIGUOUS(addContiguous, addContiguousImpl)
FOR_ALL_TYPES_ELEMENTWISE_STRIDED(addStrided, addStridedImpl)


