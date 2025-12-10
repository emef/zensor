const std = @import("std");

const cuda = @import("cuda");
const cublas_ = cuda.cublas;

const ContextError = cuda.Error || cublas_.CublasError;

const Self = @This();

device: cuda.Device,
stream: cuda.Stream,
cublas: cublas_.Handle,

pub fn init(dev: cuda.Device) ContextError!Self {
    return Self{
        .device = dev,
        .stream = try cuda.Stream.init(dev),
        .cublas = try cublas_.Handle.init(),
    };
}

pub fn default() ContextError!Self {
    return try init(.default);
}

pub fn fork(self: Self) cuda.CudaError!Self {
    return Self{
        .device = self.device,
        .stream = try cuda.Stream.init(self.device),
        .cublas = self.cublas,
    };
}
