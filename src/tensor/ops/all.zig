const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const KAll = @import("../kernels/root.zig").KAll;
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;

pub fn all(
    ctx: Context,
    in: anytype,
) Error!bool {
    if (in.shape.elems() == 0) return true;

    return switch (in.loc()) {
        .host => return try host_all(in),
        .cuda => return try cuda_all(ctx, in),
    };
}

fn host_all(in: anytype) Error!bool {
    const in_type: DType = @TypeOf(in).dtype;
    const to_bool = DType.caster(in_type, .bool);
    var it = try in.iter();

    while (it.next()) |el| {
        if (!to_bool.cast(el)) {
            return false;
        }
    }

    return true;
}

fn cuda_all(
    ctx: Context,
    in: anytype,
) Error!bool {
    const dtype = @TypeOf(in).dtype;
    if (dtype != .bool) return error.NotImplemented;

    const device = in.loc().cuda;
    const out = try Tensor(.i32).allocDevice(device, .{1});
    defer out.deinit();

    try cuda.launchKernel(
        device,
        try KAll.init(.{
            .a = in,
            .out = out,
        }),
        ctx.stream,
    );

    try ctx.stream.sync();

    var out_buf: [1]i32 = undefined;
    try out.storage.cuda.deviceToHost(.{ .len = 1 }, &out_buf);

    return out_buf[0] != 0;
}

test all {
    const ctx = try Context.default();
    const host: Location = .{ .host = std.testing.allocator };
    const gpu: Location = .{ .cuda = ctx.device };

    // cpu-only support for non-bool types
    const ints = try Tensor(.i32).fromSlice(host, &[_]i32{ 0, 1, 2, -1 }, .{4});
    defer ints.deinit();
    const floats = try Tensor(.f32).fromSlice(host, &[_]f32{
        0, 1,
        0, -1,
    }, .{ 2, 2 });
    defer floats.deinit();

    try std.testing.expect(!try all(ctx, ints));
    @memset(ints.s(), 1);
    try std.testing.expect(try all(ctx, ints));

    try std.testing.expect(!try all(ctx, floats));
    @memset(floats.s(), 1);
    try std.testing.expect(try all(ctx, floats));

    const test_cases: []const struct {
        in: []const bool,
        expect: bool,
    } = &.{
        .{
            .in = &.{ true, true, true },
            .expect = true,
        },
        .{
            .in = &.{ false, false, false },
            .expect = false,
        },
        .{
            .in = &.{ true, true, false, true },
            .expect = false,
        },
        .{
            .in = &.{},
            .expect = true,
        },
    };

    for (&[_]Location{ host, gpu }) |loc| {
        for (test_cases) |tc| {
            var in = try Tensor(.bool).fromSlice(loc, tc.in, .{tc.in.len});
            defer in.deinit();

            try std.testing.expectEqual(tc.expect, try all(ctx, in));
        }
    }
}
