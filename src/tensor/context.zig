const std = @import("std");

const cuda = @import("cuda");
const cublas_ = cuda.cublas;

const Error = @import("root.zig").Error;

const Self = @This();

device: cuda.Device,
stream: cuda.Stream,
cublas: cublas_.Handle,

pub fn init(dev: cuda.Device) Error!Self {
    return Self{
        .device = dev,
        .stream = try cuda.Stream.init(dev),
        .cublas = try cublas_.Handle.init(),
    };
}

pub fn default() Error!Self {
    return try init(.default);
}

pub fn sync(self: Self) !void {
    return try self.stream.sync();
}

pub fn fork(self: Self) cuda.CudaError!Self {
    return Self{
        .device = self.device,
        .stream = try cuda.Stream.init(self.device),
        .cublas = self.cublas,
    };
}
