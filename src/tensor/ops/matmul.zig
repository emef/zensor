const std = @import("std");

const cuda = @import("cuda");
const cublas = cuda.cublas;
const DType = @import("core").DType;

const Context = @import("../context.zig");
const tensor = @import("../tensor.zig");
const Shape = tensor.Shape;
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;
const testing = @import("../testing.zig");

pub fn matmul(
    ctx: Context,
    lhs_in: anytype,
    rhs_in: anytype,
    comptime C: DType,
) Error!Tensor(C) {
    const A: DType = @TypeOf(lhs_in).dtype;
    const B: DType = @TypeOf(rhs_in).dtype;

    if (!lhs_in.loc().eql(rhs_in.loc())) {
        return error.WrongDevice;
    }

    var lhs: Tensor(A) = lhs_in;
    var rhs: Tensor(B) = rhs_in;

    const lhs_dims = lhs.shape.len;
    const rhs_dims = rhs.shape.len;

    if (lhs_dims == 1 and rhs_dims == 1) {
        // dot product
        return error.NotImplemented;
    }

    if (lhs_dims == 2 and rhs_dims == 1) {
        // matrix-vector product
        return error.NotImplemented;
    }

    // if both arguments are at least 1-dimensional and at least one argument is
    // N-dimensional (where N > 2), then a batched matrix multiply is returned.
    // If the first argument is 1-dimensional, a 1 is prepended to its
    // dimension for the purpose of the batched matrix multiply and removed
    // after. If the second argument is 1-dimensional, a 1 is appended to its
    // dimension for the purpose of the batched matrix multiply and removed
    // after.

    // The first N-2 dimensions of each argument, the batch dimensions, are
    // broadcast (and thus must be broadcastable). The last 2, the matrix
    // dimensions, are handled as in the matrix-matrix product.

    var transpose_lhs: bool = false;
    var transpose_rhs: bool = false;
    if (lhs_dims >= 2 and !lhs.isContiguous()) {
        if (lhs.mT().isContiguous()) {
            transpose_lhs = true;
        } else {
            return error.NotContiguous;
        }
    }
    if (rhs_dims >= 2 and !rhs.isContiguous()) {
        if (rhs.mT().isContiguous()) {
            transpose_rhs = true;
        } else {
            return error.NotContiguous;
        }
    }

    // m is rows of lhs
    // k is cols of lhs
    // k is rows of rhs
    // n is cols of rhs

    const m = if (lhs_dims == 1) 1 else lhs.shape.at(-2);
    const k = lhs.shape.at(-1);
    const n = rhs.shape.at(-1);

    if (rhs_dims > 1 and rhs.shape.at(-2) != k) {
        return error.WrongShape;
    }

    // since we are stored in row-major we need to do transpose trick:
    //   C^T = B^T A^T
    // shapes:
    //   A: (batch_a, M, K)
    //   B: (batch_b, K, N)
    //   C: (batch_out, M, N)

    var batch: i32 = 1;
    var lhs_stride: i32 = 0;
    var rhs_stride: i32 = 0;

    if (lhs_dims > 2 and rhs_dims > 2) {
        return error.NotImplemented;
    } else if (lhs_dims > 2) {
        batch = batchDim(lhs.shape);
        lhs_stride = m * k;
    } else if (rhs_dims > 2) {
        batch = batchDim(rhs.shape);
        rhs_stride = k * n;
    }

    var out = try Tensor(C).init(lhs.loc(), .{ batch, m, n });
    const out_stride = m * n;

    std.debug.print("matmul {f} x {f} (m={d} k={d} n={d}) trans_lhs={any}, trans_rhs={any}\n", .{
        lhs.shape,
        rhs.shape,
        m,
        k,
        n,
        transpose_lhs,
        transpose_rhs,
    });

    const lhs_ptr = lhs.storage.cuda.ptrOffset(rhs.offset);
    const rhs_ptr = rhs.storage.cuda.ptrOffset(lhs.offset);

    const lda = if (transpose_rhs) k else n;
    const ldb = if (transpose_lhs) m else k;

    const op = cublas.CublasGemmStridedBatch(f32, f32, f32){
        .handle = ctx.cublas, // cublas handle
        .m = n, // rows of B^T (rhs)
        .n = m, // cols of A^T (lhs)
        .k = k, // cols of B^T (out)
        .A = rhs_ptr, // A ptr (rhs)
        .transpose_a = transpose_rhs,
        .lda = lda,
        .strideA = rhs_stride, // stride of B^T (rhs)
        .B = lhs_ptr, // B ptr (lhs)
        .transpose_b = transpose_lhs,
        .ldb = ldb,
        .strideB = lhs_stride, // stride of A^T (lhs)
        .C = out.storage.cuda.ptr, // C ptr (out)
        .ldc = n, // cols of C (out)
        .strideC = out_stride, // stride of C^T (out)
        .batchCount = batch, // number of batches
        .alpha = 1,
        .beta = 0,
    };

    try op.exec(ctx.cublas, ctx.stream);

    if (lhs_dims == 1) {
        out = try out.view(.{out.shape.at(-1)});
    } else if (lhs_dims == 2) {
        if (out.shape.at(0) != 1) unreachable;
        // TODO: squeeze
        out = try out.view(.{ out.shape.at(1), out.shape.at(2) });
    }

    return out;
}

fn batchDim(sh: Shape) i32 {
    if (sh.len < 3) unreachable;
    var batch: i32 = 1;
    for (0..sh.len - 2) |i| {
        batch *= sh.data[i];
    }
    return batch;
}

test matmul {
    const ctx = try Context.default();
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const host: Location = .{ .host = std.testing.allocator };

    // zig fmt: off
    const input = try Tensor(.f32).fromSlice(
        gpu,
        &[_]f32{
            1, 2, 3,
            4, 5, 6,

            1, 1, 1,
            2, 2, 2,

            0, 0, 0,
            1, 2, 2
        },
        .{ 3, 2, 3 },
    );
    // zig fmt: on

    const weight = try Tensor(.f32).fromSlice(
        gpu,
        &[_]f32{
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        },
        .{ 3, 2 },
    );
    defer weight.deinit();

    {
        const input_1d = try input.select(.{ 0, 0 });
        try std.testing.expectEqual(1, input_1d.shape.len);

        var out = try matmul(ctx, input_1d, weight, .f32);
        defer out.deinit();

        try ctx.sync();

        try testing.expectEqual(out, &[_]f32{
            22, 28,
        });

        try std.testing.expect(out.shape.eql(.{2}));
    }

    {
        const input_2d = try input.select(.{0});
        try std.testing.expectEqual(2, input_2d.shape.len);

        var out = try matmul(ctx, input_2d, weight, .f32);
        defer out.deinit();

        try ctx.sync();

        try testing.expectEqual(out, &[_]f32{
            22, 28,
            49, 64,
        });

        try std.testing.expect(out.shape.eql(.{ 2, 2 }));
    }

    {
        try std.testing.expectEqual(3, input.shape.len);
        var out = try matmul(ctx, input, weight, .f32);
        defer out.deinit();

        try ctx.sync();

        try testing.expectEqual(out, &[_]f32{
            22, 28,
            49, 64,

            9,  12,
            18, 24,

            0,  0,
            17, 22,
        });

        try std.testing.expect(out.shape.eql(.{ 3, 2, 2 }));
    }

    // test mT
    {
        const input_mT = try Tensor(.f32).fromSlice(
            gpu,
            &[_]f32{
                1, 2,
                3, 4,
                4, 5,
                6, 7,

                8, 8,
                0, 0,
                9, 9,
                0, 0,
            },
            .{ 2, 4, 2 },
        );

        const weight_mt = weight.mT();
        try std.testing.expect(weight_mt.shape.eql(.{ 2, 3 }));

        const out = try matmul(ctx, input_mT, weight_mt, .f32);
        defer out.deinit();

        try ctx.sync();

        try std.testing.expect(out.shape.eql(.{ 2, 4, 3 }));

        const out_h = try out.copy(host);
        defer out_h.deinit();

        try testing.expectEqual(out, &[_]f32{
            5,  11, 17,
            11, 25, 39,
            14, 32, 50,
            20, 46, 72,

            24, 56, 88,
            0,  0,  0,
            27, 63, 99,
            0,  0,  0,
        });
    }

    //TODO: mixed precision
    //TODO: mixed dtypes

}
