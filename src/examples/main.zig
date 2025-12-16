const std = @import("std");

const cuda = @import("cuda");
const tensor = @import("zensor").tensor;
const Location = tensor.Location;
const Context = tensor.Context;
const Tensor = tensor.Tensor;

const MNIST = @import("mnist.zig").MNIST;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    const ctx = try Context.default();
    const host: Location = .{ .host = alloc };
    const gpu: Location = .{ .cuda = ctx.device };

    const mnist = try MNIST(.f32).init(
        alloc,
        gpu,
        "src/examples/train_py/checkpoints/mnist/epoch_5.safetensors",
    );
    defer mnist.deinit();

    std.debug.print("mnist\n", .{});

    const input = try Tensor(.f32).arange(gpu, 0, 28 * 28);
    defer input.deinit();

    var out = try mnist.forward(ctx, input);
    defer out.deinit();

    const out_h = try out.move(host);
    defer out_h.deinit();

    std.debug.print("out: {any}\n", .{out_h.s()[0..20]});
}
