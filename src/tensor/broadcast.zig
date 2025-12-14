const std = @import("std");

const Error = @import("tensor.zig").Error;
const Location = @import("tensor.zig").Location;
const Tensor = @import("tensor.zig").Tensor;

pub fn broadcastTensors(
    left_: anytype,
    right_: anytype,
) Error!struct {
    left: @TypeOf(left_),
    right: @TypeOf(right_),
} {
    // Two tensors are “broadcastable” if the following rules hold:
    // * Each tensor has at least one dimension.
    // * When iterating over the dimension sizes, starting at the
    //   trailing dimension, the dimension sizes must either be equal,
    //   one of them is 1, or one of them does not exist.

    // TODO: this is not correct (doesn't handle dimension 1 broadcast)!
    var left_out = left_;
    var right_out = right_;
    if (left_.shape.len > right_.shape.len) {
        right_out = try right_.expand(left_.shape);
    } else if (right_.shape.len > left_.shape.len) {
        left_out = try left_.expand(right_.shape);
    }

    return .{
        .left = left_out,
        .right = right_out,
    };
}

test broadcastTensors {
    const host: Location = .{ .host = std.testing.allocator };
    const left = try Tensor(.i32).fromSlice(host, &[_]i32{
        1, 2,
    }, .{2});
    defer left.deinit();
    const right = try Tensor(.f32).fromSlice(host, &[_]f32{
        1, 2,
        1, 2,
        1, 2,
        1, 2,
        1, 2,
    }, .{ 5, 2 });
    defer right.deinit();

    const bcast = try broadcastTensors(left, right);

    var left_it = try bcast.left.iter();
    var right_it = try bcast.right.iter();

    for (0..10) |_| {
        const l: i32 = left_it.next() orelse return error.TestFailed;
        const r: f32 = right_it.next() orelse return error.TestFailed;

        try std.testing.expectApproxEqAbs(
            @as(f32, @floatFromInt(l)),
            r,
            0.0001,
        );
    }
}
