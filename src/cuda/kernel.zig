const std = @import("std");

const core = @import("coretypes.zig");
const Device = @import("device.zig").Device;
const Stream = @import("stream.zig").Stream;

extern "c" fn cudaLaunchKernel(
    func: ?*const anyopaque,
    gridDim: core.Dim3,
    blockDim: core.Dim3,
    args: [*]?*anyopaque,
    sharedMem: usize,
    stream: core.cudaStream_t,
) c_int;

pub const KernelDims = struct {
    grid: core.Dim3,
    block: core.Dim3,
};

pub const max_kernel_args = 16;
pub const KernelArgs = [max_kernel_args]?*anyopaque;

pub fn kernelArgs(args: []const *anyopaque) KernelArgs {
    if (args.len > max_kernel_args) {
        @panic("too many kernel args!");
    }

    var kernel_args: KernelArgs = @splat(null);
    for (0..args.len) |i| {
        kernel_args[i] = args[i];
    }
    return kernel_args;
}

pub const KernelSpec = struct {
    func: *const anyopaque,
    dims: KernelDims,
    args: KernelArgs,
    shared_mem: ?usize = null,
};

pub fn launchKernel(
    dev: Device,
    kernel: anytype,
    stream: Stream,
) core.Error!void {
    const spec: KernelSpec = @constCast(&kernel).spec();
    var arg_ptrs = @constCast(&spec).args;

    try dev.set();
    const code = cudaLaunchKernel(
        spec.func,
        spec.dims.grid,
        spec.dims.block,
        &arg_ptrs,
        spec.shared_mem orelse 0,
        stream.ptr,
    );

    if (code != 0) {
        return error.TODO;
    }
}
