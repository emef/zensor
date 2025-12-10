const std = @import("std");

const cuda = @import("cuda");
const kernels = @import("kernels");
const Tensor = @import("tensor").Tensor;

pub fn runMMUL() !void {
    const N = 5000;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const device = cuda.device(0);
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    var h_a = try Tensor(i32).allocHost(alloc, .{ 1, N, N });
    var h_b = try Tensor(i32).allocHost(alloc, .{ 1, N, N });

    for (0..N) |i| {
        for (0..N) |j| {
            h_a.s[i * N + j] = @intCast(@mod(i * N + j, 10));
            h_b.s[i * N + j] = @intCast(@mod(i * N + j, 10));
        }
    }

    var d_a = try h_a.move(.{ .device = device });
    defer d_a.deinit();
    var d_b = try h_b.copy(.{ .device = device });
    defer d_b.deinit();
    var d_out = try Tensor(i32).allocDevice(device, .{ N, N });
    defer d_out.deinit();

    const start = try std.time.Instant.now();

    try cuda.launchKernel(
        device,
        try kernels.KMMulTiled.init(.{
            .a = d_a,
            .b = d_b,
            .out = d_out,
            .n = N,
        }),
        stream,
    );

    const h_out = try d_out.move(.{ .host = alloc });
    defer h_out.deinit();

    const us_elapsed = (try std.time.Instant.now()).since(start) / std.time.ns_per_us;

    std.debug.print("elapsed: {d}us\n", .{us_elapsed});
}
