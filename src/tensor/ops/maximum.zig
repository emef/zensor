const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const KMaximum = @import("../kernels/root.zig").KMaximum;
const tensor = @import("../root.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;
const expectTensorEqual = tensor.testing.expectEqual;

pub fn MaximumOpts(dtype: DType) type {
    return struct {
        out: ?Tensor(dtype) = null,
    };
}

pub fn maximum(
    comptime dtype: DType,
    ctx: Context,
    lhs_in: Tensor(dtype),
    rhs_in: Tensor(dtype),
    opts: MaximumOpts(dtype),
) Error!Tensor(dtype) {
    if (dtype == .bool) {
        @compileError("max not supported for bool");
    }

    if (!lhs_in.loc().eql(rhs_in.loc())) {
        return error.WrongDevice;
    }

    const bcast = try broadcastTensors(lhs_in, rhs_in);

    return switch (lhs_in.loc()) {
        .host => return try host_maximum(dtype, bcast, opts),
        .cuda => return try cuda_maximum(dtype, ctx, bcast, opts),
    };
}

fn host_maximum(
    comptime dtype: DType,
    bcast: anytype,
    opts: MaximumOpts(dtype),
) Error!Tensor(dtype) {
    const out = opts.out orelse try Tensor(dtype).empty(bcast.left.loc(), bcast.left.shape);

    if (!out.loc().eql(bcast.left.loc())) {
        return error.WrongDevice;
    }

    var left_it = try bcast.left.iter();
    var right_it = try bcast.right.iter();

    var i: usize = 0;
    while (left_it.next()) |left_val| {
        const right_val = right_it.next() orelse unreachable;
        out.s()[i] = @max(left_val, right_val);
        i += 1;
    }

    return out;
}

fn cuda_maximum(
    comptime dtype: DType,
    ctx: Context,
    bcast: anytype,
    opts: MaximumOpts(dtype),
) Error!Tensor(dtype) {
    const device = bcast.left.loc().cuda;
    const out = opts.out orelse try Tensor(dtype).empty(bcast.left.loc(), bcast.left.shape);

    if (!out.loc().eql(bcast.left.loc())) {
        return error.WrongDevice;
    }

    try cuda.launchKernel(
        device,
        try KMaximum(dtype).init(.{
            .a = bcast.left,
            .b = bcast.right,
            .out = out,
        }),
        ctx.stream,
    );

    return out;
}

test maximum {
    const ctx = try Context.default();
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const host: Location = .{ .host = std.testing.allocator };

    for (&[_]Location{ host, gpu }) |loc| {
        const lhs = try Tensor(.f32).fromSlice(loc, &[_]f32{
            1, 2, 3,
        }, .{ 1, 3 });
        defer lhs.deinit();

        const rhs = try Tensor(.f32).fromSlice(loc, &[_]f32{
            4, -1, 5,
        }, .{ 1, 3 });
        defer rhs.deinit();

        {
            const out = try maximum(.f32, ctx, lhs, rhs, .{});
            defer out.deinit();

            try std.testing.expect(out.shape.eql(.{ 1, 3 }));
            try expectTensorEqual(out, &[_]f32{ 4, 2, 5 });
        }

        // TODO: test with 0 dim tensor (scalar) when working
    }
}
