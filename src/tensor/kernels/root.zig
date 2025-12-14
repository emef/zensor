pub const KAll = @import("all.zig").KAll;
pub const KEql = @import("eql.zig").KEql;
pub const KMMulNaive = @import("mmul.zig").KMMulNaive;
pub const KMMulTiled = @import("mmul.zig").KMMulTiled;

test {
    _ = @import("mmul.zig");
    _ = @import("eql.zig");
    _ = @import("all.zig");
}
