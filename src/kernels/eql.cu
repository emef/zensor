#define DEFINE_EQL_KERNEL(suffix, type) \
      extern "C" __global__ void eqlTyped_##suffix(const type *a, const type *b, bool *out, int elems) { \
          int i = blockIdx.x * blockDim.x + threadIdx.x; \
          if (i < elems) { \
              out[i] = a[i] == b[i]; \
          } \
      }

DEFINE_EQL_KERNEL(f32, float)
DEFINE_EQL_KERNEL(f64, double)
DEFINE_EQL_KERNEL(i32, int)
DEFINE_EQL_KERNEL(u32, unsigned int)

