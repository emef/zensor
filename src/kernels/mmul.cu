extern "C" {
    __global__ void mmulNaive(int *a, int *b, int *c, int n) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;

      if (row >= n || col >= n) {
        return;
      }

      int temp = 0;
      for (int i=0; i<n; i++) {
        temp += a[row * n + i] * b[i * n + col];
      }

      c[row * n + col] = temp;
    }

    __global__ void mmulTiled(
        int *a,
        int *b,
        int *c,
        int M,
        int N,
        int P
    ) {
    extern __shared__ int cache[];

    // assert(gridDim.x == gridDim.y);
    int tiles = gridDim.x;

    int trows = blockDim.x;
    int tcols = blockDim.y;

    int *a_cache = &cache[0];
    int *b_cache = &cache[trows * tcols];

    int ti = threadIdx.x;
    int tj = threadIdx.y;


    int temp = 0;
    for (int tile = 0; tile < tiles; tile++) {
      int a_row = blockIdx.x * trows + ti;
      int a_col = tile * tcols + tj;
      int b_row = tile * trows + ti;
      int b_col = blockIdx.y * tcols + tj;

      if (a_row >= M || a_col >= N) {
        a_cache[ti * tcols + tj] = 0;
      } else {
        a_cache[ti * tcols + tj] = a[a_row * N + a_col];
      }

      if (b_row >= N || b_col >= P) {
        b_cache[tj * trows + ti] = 0;
      } else {
        b_cache[tj * trows + ti] = b[b_row * P + b_col];
      }

      __syncthreads();

      for (int i=0; i<trows; i++) {
        temp += a_cache[ti * tcols + i] * b_cache[tj * trows + i];
      }
    }

    // output coords
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    int cj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ci >= M || cj >= P) return;

    c[ci * P + cj] = temp;
  }
}
