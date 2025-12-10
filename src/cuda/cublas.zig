const core = @import("coretypes.zig");
const Stream = @import("stream.zig").Stream;

pub const CublasError = error{
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
};

const cublasHandle_t = ?*anyopaque;

const cublasStatus_t = enum(c_int) {
    success = 0,
    not_initialized = 1,
    alloc_failed = 3,
    invalid_value = 7,
    arch_mismatch = 8,
    mapping_error = 11,
    execution_failed = 13,
    internal_error = 14,
    not_supported = 15,
    license_error = 16,
};

const cublasOperation_t = enum(c_int) {
    n = 0,
    t = 1,
    c = 2,
};

const cublasComputeType_t = enum(c_int) {
    compute_16f = 64,
    compute_16f_pedantic = 65,
    compute_32f = 68,
    compute_32f_pedantic = 69,
    compute_32f_fast_16f = 74,
    compute_32f_fast_16bf = 75,
    compute_32f_fast_tf32 = 77,
    compute_64f = 70,
    compute_64f_pedantic = 71,
    compute_32i = 72,
    compute_32i_pedantic = 73,
};

const cublasGemmAlgo_t = enum(c_int) {
    default = -1,
    algo0 = 0,
    algo1 = 1,
    algo2 = 2,
    algo3 = 3,
    algo4 = 4,
    algo5 = 5,
    algo6 = 6,
    algo7 = 7,
    algo8 = 8,
    algo9 = 9,
    algo10 = 10,
    algo11 = 11,
    algo12 = 12,
    algo13 = 13,
    algo14 = 14,
    algo15 = 15,
    algo16 = 16,
    algo17 = 17,
    algo18 = 18,
    algo19 = 19,
    algo20 = 20,
    algo21 = 21,
    algo22 = 22,
    algo23 = 23,
    default_tensor_op = 99,
    algo0_tensor_op = 100,
    algo1_tensor_op = 101,
    algo2_tensor_op = 102,
    algo3_tensor_op = 103,
    algo4_tensor_op = 104,
    algo5_tensor_op = 105,
    algo6_tensor_op = 106,
    algo7_tensor_op = 107,
    algo8_tensor_op = 108,
    algo9_tensor_op = 109,
    algo10_tensor_op = 110,
    algo11_tensor_op = 111,
    algo12_tensor_op = 112,
    algo13_tensor_op = 113,
    algo14_tensor_op = 114,
    algo15_tensor_op = 115,
};

extern fn cublasCreate_v2(handle: *cublasHandle_t) cublasStatus_t;
extern fn cublasSetStream_v2(handle: cublasHandle_t, stream: core.cudaStream_t) cublasStatus_t;

pub const Handle = struct {
    ptr: cublasHandle_t,

    pub fn init() CublasError!Handle {
        var ptr: cublasHandle_t = undefined;

        const status = cublasCreate_v2(&ptr);
        try checkCublas(status);

        return Handle{
            .ptr = ptr,
        };
    }

    pub fn deinit(self: Handle) void {
        _ = self;
        // TODO;
    }
};

fn checkCublas(status: cublasStatus_t) !void {
    switch (status) {
        .success => return,
        .not_initialized => return error.NotInitialized,
        .alloc_failed => return error.AllocFailed,
        .invalid_value => return error.InvalidValue,
        .arch_mismatch => return error.ArchMismatch,
        .mapping_error => return error.MappingError,
        .execution_failed => return error.ExecutionFailed,
        .internal_error => return error.InternalError,
        .not_supported => return error.NotSupported,
        .license_error => return error.LicenseError,
    }
}

extern fn cublasGemmStridedBatchedEx(
    handle: ?*const anyopaque,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const anyopaque,
    A: ?*const anyopaque,
    Atype: core.cudaDataType_t,
    lda: c_int,
    strideA: c_longlong,
    B: ?*const anyopaque,
    Btype: core.cudaDataType_t,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const anyopaque,
    C: ?*anyopaque,
    Ctype: core.cudaDataType_t,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) cublasStatus_t;

pub fn CublasGemmStridedBatch(A: type, B: type, C: type) type {
    const computeType: cublasComputeType_t = comptime blk: {
        if (A == f16 and B == f16 and C == f16) break :blk .compute_16f;
        if (A == i8 and B == i8 and C == i32) break :blk .compute_32i;
        if (A == f32 and B == f32 and C == f32) break :blk .compute_32f_fast_16bf;
        if (A == f64 and B == f64 and C == f64) break :blk .compute_64f;
        if (A == f16 and B == f16 and C == f32) break :blk .compute_32f;
        if (A == i8 and B == f16 and C == f32) break :blk .compute_32f;
        if (A == i8 and B == f32 and C == f32) break :blk .compute_32f;
        if (A == f16 and B == i8 and C == f32) break :blk .compute_32f;
        if (A == f32 and B == i8 and C == f32) break :blk .compute_32f;
        if (A == f16 and B == f32 and C == f32) break :blk .compute_32f;
        if (A == f32 and B == f16 and C == f32) break :blk .compute_32f;
        @compileError("Unsupported type combination for cuBLAS compute type");
    };

    const scaleType: type = switch (computeType) {
        .compute_16f => f16,
        .compute_32i => i32,
        .compute_32f => f32,
        .compute_32f_fast_16bf => f32,
        .compute_64f => f64,
        else => unreachable,
    };

    return struct {
        const Self = @This();
        handle: Handle,
        m: i32,
        n: i32,
        k: i32,
        A: ?*const anyopaque,
        lda: c_int,
        strideA: c_longlong,
        B: ?*const anyopaque,
        ldb: c_int,
        strideB: c_longlong,
        C: ?*anyopaque,
        ldc: c_int,
        strideC: c_longlong,
        batchCount: c_int,
        alpha: f64 = 1,
        beta: f64 = 0,

        pub fn exec(self: Self, handle: Handle, stream: Stream) CublasError!void {
            const alpha: scaleType = 1;
            const beta: scaleType = 0;

            try checkCublas(cublasSetStream_v2(handle.ptr, stream.ptr));

            const status = cublasGemmStridedBatchedEx(
                self.handle.ptr,
                .n,
                .n,
                @intCast(self.m),
                @intCast(self.n),
                @intCast(self.k),
                &alpha,
                self.A,
                core.cudaDataType_t.fromType(A),
                @intCast(self.lda),
                @intCast(self.strideA),
                self.B,
                core.cudaDataType_t.fromType(B),
                @intCast(self.ldb),
                @intCast(self.strideB),
                &beta,
                self.C,
                core.cudaDataType_t.fromType(C),
                @intCast(self.ldc),
                @intCast(self.strideC),
                @intCast(self.batchCount),
                computeType,
                .default,
            );

            try checkCublas(status);
        }
    };
}
