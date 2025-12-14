const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;
const tensor = @import("tensor");
const Tensor = tensor.Tensor;
const Location = tensor.Location;
const Shape = tensor.Shape;
const Context = tensor.Context;
const Error = tensor.Error;
const expectTensorEqual = tensor.testing.expectEqual;

pub fn Linear(dtype: DType) type {
    return struct {
        const Self = @This();

        weight: Tensor(dtype),
        bias: ?Tensor(dtype),

        pub const Opts = struct {
            in_features: usize,
            out_features: usize,
            bias: bool = true,
        };

        pub fn init(loc: Location, opts: Opts) Error!Self {
            const w_shape = Shape.init(.{ opts.out_features, opts.in_features });
            const b_shape = Shape.init(.{opts.out_features});

            return Self{
                .weight = try Tensor(dtype).init(loc, w_shape),
                .bias = if (opts.bias) try Tensor(dtype).init(loc, b_shape) else null,
            };
        }

        pub fn deinit(self: Self) void {
            self.weight.deinit();
            if (self.bias) |b| b.deinit();
        }

        pub fn forward(self: Self, ctx: Context, x: Tensor(dtype)) Error!Tensor(dtype) {
            if (x.shape.at(-1) != self.weight.shape.at(1)) {
                std.debug.print(
                    "expected last dim {d}, found {d}\n",
                    .{
                        x.shape.at(-1),
                        self.weight.shape.at(0),
                    },
                );
                return error.WrongShape;
            }

            // (batch, d_in)
            const x_in = try x.view(.{ -1, x.shape.at(-1) });

            var res = try x_in.matmul(ctx, self.weight.mT());
            if (self.bias) |b| {
                res = try res.add(ctx, b);
            }

            // fix output shape to match input shape (besides out dim)
            return try res.view(x.shape.replace(-1, res.shape.at(-1)));
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

    try std.testing.expect(lin.weight.shape.eql(.{ 20, 10 }));

    if (lin.bias) |b| {
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
}

test "Linear forward" {
    const ctx = try Context.default();
    const gpu: Location = .{ .cuda = ctx.device };
    const host: Location = .{ .host = std.testing.allocator };
    const lin = try Linear(.f32).init(gpu, .{
        .in_features = 2,
        .out_features = 4,
        .bias = false,
    });
    defer lin.deinit();

    var weight_buf = [8]f32{
        1, 3,
        5, 7,
        2, 4,
        6, 8,
    };

    try lin.weight.storage.cuda.hostToDevice(&weight_buf);

    const x_batch = try Tensor(.f32).fromSlice(gpu, &[_]f32{
        0.5, 1.5,
        3.5, 4.5,

        0.5, 1.5,
        3.5, 4.5,

        0.5, 1.5,
        3.5, 4.5,
    }, .{ 3, 2, 2 });

    defer x_batch.deinit();

    const fwd = try lin.forward(ctx, x_batch);
    defer fwd.deinit();

    try ctx.sync();

    std.debug.print("fwd shape {f}\n", .{fwd.shape});
    try std.testing.expect(fwd.shape.eql(.{ 3, 2, 4 }));

    const fwd_h = try fwd.copy(host);
    defer fwd_h.deinit();

    std.debug.print("fwd out: {any}\n", .{fwd_h.s()});

    try expectTensorEqual(fwd, &[_]f32{
        5,  13, 7,  15,
        17, 49, 25, 57,

        5,  13, 7,  15,
        17, 49, 25, 57,

        5,  13, 7,  15,
        17, 49, 25, 57,
    });
}
