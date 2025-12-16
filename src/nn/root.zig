const std = @import("std");

const layers = @import("layers/root.zig");
pub const Linear = layers.Linear;

test {
    std.testing.refAllDecls(@This());
}
