const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;
const tensor = @import("tensor");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Shape = tensor.Shape;

const Context = @import("context.zig");

pub fn Linear(dtype: DType) type {
    return struct {
        const Self = @This();

        a: Tensor(dtype),
        b: ?Tensor(dtype),

        pub const Opts = struct {
            in_features: usize,
            out_features: usize,
            bias: bool = true,
        };

        pub fn init(loc: Location, opts: Opts) !Self {
            const a_shape = Shape.init(.{ opts.in_features, opts.out_features });
            switch (loc) {
                .host => |alloc| {
                    const a = try Tensor(dtype).allocHost(alloc, a_shape);
                    var b: ?Tensor(dtype) = null;
                    if (opts.bias) {
                        b = try Tensor(dtype).allocHost(alloc, .{opts.out_features});
                    }
                    return Self{ .a = a, .b = b };
                },
                .cuda => |device| {
                    const a = try Tensor(dtype).allocDevice(device, a_shape);
                    var b: ?Tensor(dtype) = null;
                    if (opts.bias) {
                        b = try Tensor(dtype).allocDevice(device, .{opts.out_features});
                    }
                    return Self{ .a = a, .b = b };
                },
            }
        }

        pub fn deinit(self: Self) void {
            self.a.deinit();
            if (self.b) |b| b.deinit();
        }

        pub fn forward(self: Self, ctx: Context, x: Tensor(dtype)) !Tensor(dtype) {
            if (!self.a.loc().eql(x.loc())) {
                return error.WrongDevice;
            }

            if (self.b) |b| {
                if (!b.loc().eql(x.loc())) {
                    return error.WrongDevice;
                }
            }

            if (x.shape.at(-1) != self.a.shape.at(0)) {
                std.debug.print(
                    "expected last dim {d}, found {d}\n",
                    .{
                        x.shape.at(-1),
                        self.a.shape.at(0),
                    },
                );
                return error.WrongShape;
            }

            switch (x.loc()) {
                .host => {
                    return error.NotImplemented;
                },
                .cuda => {
                    return self.forwardCuda(ctx, x);
                },
            }
        }

        fn forwardCuda(self: Self, ctx: Context, x: Tensor(dtype)) !Tensor(dtype) {
            // (batch, d_in)
            const x_in = try x.view(.{ -1, x.shape.at(-1) });
            _ = x_in;
            _ = self;
            _ = ctx;
            return error.NotImplemented;
        }
    };
}

test "Linear shape" {
    const lin = try Linear(.f32).init(.{ .host = std.testing.allocator }, .{
        .in_features = 10,
        .out_features = 20,
        .bias = true,
    });
    defer lin.deinit();

    try std.testing.expect(lin.a.shape.eql(.{ 10, 20 }));

    if (lin.b) |b| {
        try std.testing.expect(b.shape.eql(.{20}));
    } else {
        return error.TestUnexpectedResult;
    }
}

test "Linear forward validation" {
    const ctx: Context = undefined;
    const device = cuda.device(0);
    const lin = try Linear(.f32).init(.{ .cuda = device }, .{
        .in_features = 10,
        .out_features = 20,
        .bias = true,
    });
    defer lin.deinit();

    const x = try Tensor(.f32).allocDevice(device, .{11});
    defer x.deinit();

    try std.testing.expectError(error.WrongShape, lin.forward(ctx, x));

    const x_host = try Tensor(.f32).allocHost(std.testing.allocator, .{10});
    defer x_host.deinit();

    try std.testing.expectError(error.WrongDevice, lin.forward(ctx, x_host));
}
