const tensor = @import("../root.zig");

pub const TensorInfo = extern struct {
    strides: [tensor.max_dims]c_int,
    ndim: c_int,
};

pub const KernelArgs = extern struct {
    shape: [tensor.max_dims]c_int,
    ndim: c_int,
    elems: c_int,
};
