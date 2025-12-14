const std = @import("std");

pub const Context = @import("context.zig");
pub const mmul = @import("ops/mmul.zig").mmul;
pub const ops = @import("ops/root.zig");
const tensor = @import("tensor.zig");
pub const max_dims = tensor.max_dims;
pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const Strides = tensor.Strides;
pub const HostError = tensor.HostError;
pub const HostMem = tensor.HostMem;
pub const Location = tensor.Location;
pub const Storage = tensor.Storage;
pub const Error = tensor.Error;
pub const TensorIterator = @import("iterator.zig").TensorIterator;
pub const testing = @import("testing.zig");

test {
    const broadcast = @import("broadcast.zig");
    // const ops = @import("ops/root.zig");
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(broadcast);
    std.testing.refAllDecls(ops);
}
