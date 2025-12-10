const std = @import("std");

const cuda = @import("cuda");
const cublas = cuda.cublas;
const DType = @import("core").DType;

const Context = @import("../context.zig");
const tensor = @import("../tensor.zig");
const Shape = tensor.Shape;
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const testing = @import("../testing.zig");

pub fn mmul(
    ctx: Context,
    lhs_in: anytype,
    rhs_in: anytype,
    comptime C: DType,
) !Tensor(C) {
    const A: DType = @TypeOf(lhs_in).dtype;
    const B: DType = @TypeOf(rhs_in).dtype;

    if (!lhs_in.loc().eql(rhs_in.loc())) {
        return error.WrongDevice;
    }

    const lhs_dims = lhs_in.shape.len;
    const rhs_dims = rhs_in.shape.len;

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

    var lhs: Tensor(A) = undefined;
    var rhs: Tensor(B) = undefined;

    // broadcast lhs to (batch, M, K)
    if (lhs_dims == 1) {
        lhs = try lhs_in.view(.{ 1, 1, lhs_in.shape.at(0) });
    } else {
        lhs = try lhs_in.view(.{ -1, lhs_in.shape.at(-2), lhs_in.shape.at(-1) });
    }

    // broadcast rhs to (batch, K, N)
    if (rhs_dims == 1) {
        rhs = try rhs_in.view(.{ 1, 1, rhs_in.shape.at(0) });
    } else {
        rhs = try rhs_in.view(.{ -1, rhs_in.shape.at(-2), rhs_in.shape.at(-1) });
    }

    const m: i32 = @intCast(lhs.shape.at(1));
    const k: i32 = @intCast(lhs.shape.at(2));
    const n: i32 = @intCast(rhs.shape.at(2));

    if (rhs.shape.at(1) != k) {
        return error.WrongSize;
    }

    const out_batch = lhs.shape.at(0) * rhs.shape.at(0);
    var out = try Tensor(C).init(lhs.loc(), .{ out_batch, m, n });
    const out_stride = m * n;

    // since we are stored in row-major we need to do transpose trick:
    //   C^T = B^T A^T
    // shapes:
    //   A: (batch_a, M, K)
    //   B: (batch_b, K, N)
    //   C: (batch_out, M, N)

    var batch: i32 = 1;
    var lhs_stride: i32 = 0;
    var rhs_stride: i32 = 0;
    if (lhs.shape.at(0) == 1) {
        batch = @intCast(rhs.shape.at(0));
        rhs_stride = @intCast(rhs.shape.at(1) * rhs.shape.at(2));
    } else if (rhs.shape.at(0) == 1) {
        batch = @intCast(lhs.shape.at(0));
        lhs_stride = @intCast(lhs.shape.at(1) * lhs.shape.at(2));
    } else {
        return error.NotImplemented;
    }

    // TODO: optimize mT tensors without calling .contiguous()
    if (!lhs.isContiguous()) {
        // TODO: call .contiguous()
        return error.NotImplemented;
    }
    if (!rhs.isContiguous()) {
        // TODO: call .contiguous()
        return error.NotImplemented;
    }

    const lhs_ptr = lhs.storage.cuda.ptrOffset(rhs.offset);
    const rhs_ptr = rhs.storage.cuda.ptrOffset(lhs.offset);

    const op = cublas.CublasGemmStridedBatch(f32, f32, f32){
        .handle = ctx.cublas, // cublas handle
        .m = n, // rows of B^T (rhs)
        .n = m, // cols of A^T (lhs)
        .k = k, // cols of B^T (out)
        .A = rhs_ptr, // A ptr (rhs)
        .lda = n, // cols of B (rhs)
        .strideA = rhs_stride, // stride of B^T (rhs)
        .B = lhs_ptr, // B ptr (lhs)
        .ldb = k, // cols of A (lhs)
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
    }

    return out;
}

test mmul {
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
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        },
        .{ 3, 2 },
    );
    defer weight.deinit();

    {
        const input_1d = try input.select(.{ 0, 0 });
        try std.testing.expectEqual(1, input_1d.shape.len);

        var out = try mmul(ctx, input_1d, weight, .f32);
        out = try out.move(host);
        defer out.deinit();

        try testing.expectEqual(out, &[_]f32{
            1.0, 2.0,
        });
    }

    {
        const input_2d = try input.select(.{0});
        try std.testing.expectEqual(2, input_2d.shape.len);

        var out = try mmul(ctx, input_2d, weight, .f32);
        out = try out.move(host);
        defer out.deinit();

        try testing.expectEqual(out, &[_]f32{
            1.0, 2.0,
            4.0, 5.0,
        });
    }

    {
        try std.testing.expectEqual(3, input.shape.len);
        var out = try mmul(ctx, input, weight, .f32);

        out = try out.move(host);
        defer out.deinit();

        try testing.expectEqual(out, &[_]f32{
            1.0, 2.0,
            4.0, 5.0,

            1.0, 1.0,
            2.0, 2.0,

            0.0, 0.0,
            1.0, 2.0,
        });
    }

    //TODO: mixed precision
}
