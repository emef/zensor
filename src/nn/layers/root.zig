const std = @import("std");

pub const Linear = @import("linear.zig").Linear;

test {
    std.testing.refAllDecls(@This());
}
