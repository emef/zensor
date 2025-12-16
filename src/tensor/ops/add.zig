const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const KAdd = @import("../kernels/root.zig").KAdd;
const tensor = @import("../root.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;
const expectTensorEqual = tensor.testing.expectEqual;

pub fn AddOpts(dtype: DType) type {
    return struct {
        alpha: f32 = 1.0,
        out: ?Tensor(dtype) = null,
    };
}

pub fn add(
    comptime dtype: DType,
    ctx: Context,
    lhs_in: Tensor(dtype),
    rhs_in: Tensor(dtype),
    opts: AddOpts(dtype),
) Error!Tensor(dtype) {
    if (!lhs_in.loc().eql(rhs_in.loc())) {
        return error.WrongDevice;
    }

    if (opts.alpha != 1) {
        return error.NotImplemented;
    }

    const bcast = try broadcastTensors(lhs_in, rhs_in);

    return switch (lhs_in.loc()) {
        .host => return try host_add(dtype, bcast, opts),
        .cuda => return try cuda_add(dtype, ctx, bcast, opts),
    };
}

fn host_add(comptime dtype: DType, bcast: anytype, opts: AddOpts(dtype)) Error!Tensor(dtype) {
    const out = opts.out orelse try Tensor(dtype).empty(bcast.left.loc(), bcast.left.shape);

    if (!out.loc().eql(bcast.left.loc())) {
        return error.WrongDevice;
    }

    var left_it = try bcast.left.iter();
    var right_it = try bcast.right.iter();

    var i: usize = 0;
    while (left_it.next()) |left_val| {
        const right_val = right_it.next() orelse unreachable;
        out.s()[i] = left_val + right_val;
        i += 1;
    }

    return out;
}

fn cuda_add(
    comptime dtype: DType,
    ctx: Context,
    bcast: anytype,
    opts: AddOpts(dtype),
) Error!Tensor(dtype) {
    const device = bcast.left.loc().cuda;
    const out = opts.out orelse try Tensor(dtype).empty(bcast.left.loc(), bcast.left.shape);

    if (!out.loc().eql(bcast.left.loc())) {
        return error.WrongDevice;
    }

    try cuda.launchKernel(
        device,
        try KAdd(dtype).init(.{
            .a = bcast.left,
            .b = bcast.right,
            .out = out,
        }),
        ctx.stream,
    );

    return out;
}

test add {
    const ctx = try Context.default();
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const host: Location = .{ .host = std.testing.allocator };

    for (&[_]Location{ host, gpu }) |loc| {
        const lhs = try Tensor(.f32).fromSlice(loc, &[_]f32{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,

            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
        }, .{ 2, 2, 5 });
        defer lhs.deinit();

        const rhs = try Tensor(.f32).fromSlice(loc, &[_]f32{
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
        }, .{ 2, 5 });
        defer rhs.deinit();

        {
            const added = try add(.f32, ctx, lhs, rhs, .{});
            defer added.deinit();

            try std.testing.expect(added.shape.eql(.{ 2, 2, 5 }));
            try expectTensorEqual(added, &[_]f32{
                12, 14, 16, 18, 20,
                22, 24, 26, 28, 30,

                12, 14, 16, 18, 20,
                22, 24, 26, 28, 30,
            });
        }

        {
            const added = try add(.f32, ctx, try lhs.select(0), rhs, .{});
            defer added.deinit();

            try std.testing.expect(added.shape.eql(.{ 2, 5 }));
            try expectTensorEqual(added, &[_]f32{
                12, 14, 16, 18, 20,
                22, 24, 26, 28, 30,
            });
        }

        {
            const added = try add(.f32, ctx, try lhs.select(.{ 0, 0 }), rhs, .{});
            defer added.deinit();

            try std.testing.expect(added.shape.eql(.{ 2, 5 }));
            try expectTensorEqual(added, &[_]f32{
                12, 14, 16, 18, 20,
                17, 19, 21, 23, 25,
            });
        }

        {
            const added = try add(.f32, ctx, try lhs.select(.{ 0, 0, 4 }), rhs, .{});
            defer added.deinit();

            try std.testing.expect(added.shape.eql(.{ 2, 5 }));
            try expectTensorEqual(added, &[_]f32{
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
            });
        }
    }
}
