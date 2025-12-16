const std = @import("std");

pub const F = @import("functional/root.zig");
const layers = @import("layers/root.zig");
pub const Linear = layers.Linear;

test {
    std.testing.refAllDecls(@This());
}
