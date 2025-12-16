const std = @import("std");

const zensor = @import("zensor");
const tensor = zensor.tensor;
const SafeTensors = tensor.SafeTensors;
const nn = zensor.nn;
const DType = zensor.DType;
const Tensor = tensor.Tensor;
const Error = tensor.Error;
const Context = tensor.Context;
const Location = tensor.Location;

pub fn MNIST(dtype: DType) type {
    return struct {
        const Self = @This();

        linear_1: nn.Linear(dtype),

        pub fn init(
            alloc: std.mem.Allocator,
            loc: Location,
            path: []const u8,
        ) !Self {
            const tensors = try SafeTensors.mmap(alloc, path);
            defer tensors.deinit();

            return Self{
                .linear_1 = nn.Linear(dtype){
                    .weight = try tensors.get(dtype, loc, "layers.0.weight"),
                    .bias = try tensors.get(dtype, loc, "layers.0.bias"),
                },
            };
        }

        pub fn deinit(self: Self) void {
            self.linear_1.deinit();
        }

        pub fn forward(self: Self, ctx: Context, x: Tensor(dtype)) !Tensor(dtype) {
            const out = try self.linear_1.forward(ctx, x);
            return out;
        }
    };
}
