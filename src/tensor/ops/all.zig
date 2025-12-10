const std = @import("std");

const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;

pub fn all(
    ctx: Context,
    in: anytype,
) Error!bool {
    return switch (in.loc()) {
        .host => return try host_all(in),
        .cuda => return try cuda_all(ctx, in),
    };
}

fn host_all(in: anytype) Error!bool {
    const in_type: DType = @TypeOf(in).dtype;
    const to_bool = DType.caster(in_type, .bool);
    var it = try in.iter();

    var all_true = true;
    while (it.next()) |el| {
        all_true = all_true and to_bool.cast(el);
    }

    return all_true;
}

fn cuda_all(
    ctx: Context,
    bcast: anytype,
) Error!bool {
    _ = ctx;
    _ = bcast;
    return error.NotImplemented;
}

test all {
    const ctx = try Context.default();
    const host: Location = .{ .host = std.testing.allocator };
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
}
