const std = @import("std");

const Context = @import("context.zig");
const Location = @import("root.zig").Location;

pub fn expectEqual(lhs: anytype, rhs_in: anytype) !void {
    const ctx = try Context.default();

    const type_info = @typeInfo(@TypeOf(rhs_in));

    var rhs: @TypeOf(lhs) = undefined;
    if (type_info == .pointer) {
        rhs = try @TypeOf(lhs).fromSlice(lhs.loc(), rhs_in, lhs.shape);
    } else {
        rhs = try rhs.view(rhs.shape);
    }
    defer rhs.deinit();

    const eql = try lhs.eql(ctx, rhs);
    defer eql.deinit();

    if (try eql.all(ctx)) {
        return;
    }

    if (lhs.shape.elems() < 20 and rhs.shape.elems() < 20) {
        const host: Location = .{ .host = std.testing.allocator };
        const lhs_h = try lhs.copy(host);
        defer lhs_h.deinit();
        const rhs_h = try rhs.copy(host);
        defer rhs_h.deinit();
        std.debug.print("{any} != {any}\n", .{ lhs_h.s(), rhs_h.s() });
    }

    return error.TestFailed;
}
