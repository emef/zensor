const std = @import("std");

const Context = @import("context.zig");

pub fn expectEqual(lhs: anytype, rhs_in: anytype) !void {
    const ctx = try Context.default();

    const type_info = @typeInfo(@TypeOf(rhs_in));

    var rhs: @TypeOf(lhs) = undefined;
    if (type_info == .pointer) {
        rhs = try @TypeOf(lhs).fromSlice(lhs.loc(), rhs_in, lhs.shape);
    } else {
        rhs = rhs.view(rhs.shape);
    }
    defer rhs.deinit();

    const eql = try lhs.eql(ctx, rhs);
    defer eql.deinit();

    return try std.testing.expect(try eql.all(ctx));
}
