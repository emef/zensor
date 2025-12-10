const std = @import("std");

const DType = @import("core").DType;

const broadcastTensors = @import("../broadcast.zig").broadcastTensors;
const Context = @import("../context.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Error = tensor.Error;

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
    _ = ctx;
    _ = bcast;
    return error.NotImplemented;
}

test eql {
    const ctx = try Context.default();
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

    {
        const eql_tensor = try eql(ctx, left, right);
        defer eql_tensor.deinit();

        var eql_it = try eql_tensor.iter();
        while (eql_it.next()) |el| {
            try std.testing.expect(el);
        }
    }

    {
        // the even indices are all wrong now
        left.s()[0] = 0;

        const eql_tensor = try eql(ctx, left, right);
        defer eql_tensor.deinit();

        var eql_it = try eql_tensor.iter();
        var i: usize = 0;
        while (eql_it.next()) |el| : (i += 1) {
            const expect_eq = i % 2 == 1;
            try std.testing.expectEqual(expect_eq, el);
        }
    }
}
