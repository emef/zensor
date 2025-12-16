#include <cstdint>
#include "common.cuh"

template <typename T>
 __device__ __forceinline__ void maximumContiguousImpl(
    const T *a,
    const T *b,
    T *out,
    int i
) {
    T a_ = a[i];
    T b_ = b[i];
    out[i] = a_ > b_ ? a_ : b_;
}

template <typename T>
__device__ void maximumStridedImpl(
    const T *a, TensorInfo a_info,
    const T *b, TensorInfo b_info,
    T *out,
    KernelArgs args,
    int i
) {
    int a_off = stridedOffset(i, args.shape, a_info.strides, args.ndim);
    int b_off = stridedOffset(i, args.shape, b_info.strides, args.ndim);
    T a_ = a[a_off];
    T b_ = b[b_off];
    out[i] = a_ > b_ ? a_ : b_;
}

FOR_ALL_TYPES_ELEMENTWISE_CONTIGUOUS(maximumContiguous, maximumContiguousImpl)
FOR_ALL_TYPES_ELEMENTWISE_STRIDED(maximumStrided, maximumStridedImpl)


