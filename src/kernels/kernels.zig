pub const KMMulNaive = @import("mmul.zig").KMMulNaive;
pub const KMMulTiled = @import("mmul.zig").KMMulTiled;

test {
    _ = @import("mmul.zig");
    _ = @import("eql.zig");
}
