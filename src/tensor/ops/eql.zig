const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const KEql = @import("../kernels/root.zig").KEql;
const tensor = @import("../root.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;
const expectTensorEqual = tensor.testing.expectEqual;

pub fn eql(
    ctx: Context,
    lhs_in: anytype,
    rhs_in: anytype,
) Error!Tensor(.bool) {
    if (!lhs_in.loc().eql(rhs_in.loc())) {
        return error.WrongDevice;
    }

    const bcast = try broadcastTensors(lhs_in, rhs_in);

    return switch (lhs_in.loc()) {
        .host => return host_eql(bcast),
        .cuda => return try cuda_eql(ctx, bcast),
    };
}

fn host_eql(bcast: anytype) Error!Tensor(.bool) {
    const lhs = bcast.left;
    const rhs = bcast.right;
    const out = try Tensor(.bool).init(lhs.loc(), lhs.shape);
    var slice: []bool = out.storage.host.slice;

    var left_it = try lhs.iter();
    var right_it = try rhs.iter();

    const lhs_dtype: DType = @TypeOf(lhs).dtype;
    const rhs_dtype: DType = @TypeOf(rhs).dtype;
    const dest_type: DType = comptime DType.promote(lhs_dtype, rhs_dtype);
    const lhs_promoter = DType.caster(lhs_dtype, dest_type);
    const rhs_promoter = DType.caster(rhs_dtype, dest_type);

    for (0..slice.len) |i| {
        const l = left_it.next() orelse unreachable;
        const r = right_it.next() orelse unreachable;

        slice[i] = lhs_promoter.cast(l) == rhs_promoter.cast(r);
    }

    return out;
}

fn cuda_eql(
    ctx: Context,
    bcast: anytype,
) Error!Tensor(.bool) {
    const device = bcast.left.loc().cuda;
    const dtype = @TypeOf(bcast.left).dtype;
    const shape = bcast.left.shape;

    if (@TypeOf(bcast.right).dtype != dtype) {
        return error.WrongType;
    }

    const out = try Tensor(.bool).allocDevice(device, shape);

    try cuda.launchKernel(
        device,
        try KEql(dtype).init(.{
            .a = bcast.left,
            .b = bcast.right,
            .out = out,
        }),
        ctx.stream,
    );

    return out;
}

test eql {
    const ctx = try Context.default();
    const host: Location = .{ .host = std.testing.allocator };
    const gpu: Location = .{ .cuda = cuda.device(0) };

    const test_cases: []const struct {
        left: []const i32,
        right: []const i32,
        expect: ?[]const bool = null,
        expect_err: bool = false,
    } = &.{
        .{
            .left = &.{
                1, 2,
            },
            .right = &.{
                1, 2,
                1, 2,
                1, 2,
            },
            .expect = &.{
                true, true,
                true, true,
                true, true,
            },
        },

        .{
            .left = &.{
                1, 2,
            },
            .right = &.{
                1, 2,
                0, 2,
                1, 20,
                8, 8,
            },
            .expect = &.{
                true,  true,
                false, true,
                true,  false,
                false, false,
            },
        },

        .{
            .left = &.{
                1, 2,
            },
            .right = &.{
                1, 2,
                0, 2,
                1, 20,
                8, 8,
            },
            .expect = &.{
                true,  true,
                false, true,
                true,  false,
                false, false,
            },
        },
    };

    for (&[_]Location{ host, gpu }) |loc| {
        for (test_cases) |tc| {
            var left = try Tensor(.i32).fromSlice(loc, tc.left, .{tc.left.len});
            defer left.deinit();
            var right = try Tensor(.i32).fromSlice(loc, tc.right, .{tc.right.len});
            defer right.deinit();

            var eql_tensor = eql(
                ctx,
                // TODO: try left.view(.{ -1, 2 }),
                try left.view(.{2}),
                try right.view(.{ -1, 2 }),
            ) catch {
                if (!tc.expect_err) {
                    return error.TestFailed;
                }
                continue;
            };
            defer eql_tensor.deinit();

            try ctx.stream.sync();

            const eql_h = try eql_tensor.move(host);
            defer eql_h.deinit();

            const expect = tc.expect orelse return error.TestFailed;

            try std.testing.expectEqual(eql_h.shape.elems(), expect.len);
            for (0..expect.len) |i| {
                try std.testing.expectEqual(expect[i], eql_h.s()[i]);
            }
        }
    }
}
