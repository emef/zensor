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
        linear_2: nn.Linear(dtype),
        linear_3: nn.Linear(dtype),
        relu: nn.F.ReLU(dtype),

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
                .linear_2 = nn.Linear(dtype){
                    .weight = try tensors.get(dtype, loc, "layers.3.weight"),
                    .bias = try tensors.get(dtype, loc, "layers.3.bias"),
                },
                .linear_3 = nn.Linear(dtype){
                    .weight = try tensors.get(dtype, loc, "layers.6.weight"),
                    .bias = try tensors.get(dtype, loc, "layers.6.bias"),
                },
                .relu = try nn.F.ReLU(dtype).init(loc),
            };
        }

        pub fn deinit(self: Self) void {
            self.linear_1.deinit();
        }

        pub fn forward(self: Self, ctx: Context, x: Tensor(dtype)) !Tensor(dtype) {
            if (x.shape.len < 2) {
                return error.WrongShape;
            }

            var out = try x.flattenDims(1, -1);
            out = try self.linear_1.forward(ctx, out);
            out = try self.relu.apply(ctx, out);
            out = try self.linear_2.forward(ctx, out);
            out = try self.relu.apply(ctx, out);
            out = try self.linear_3.forward(ctx, out);
            return out;
        }
    };
}
