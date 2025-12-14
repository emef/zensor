const std = @import("std");

const Context = @import("tensor").Context;
const cuda = @import("cuda");
const eql = @import("tensor").ops.eql;
const kernels = @import("kernels");
const Location = @import("tensor").Location;
const Tensor = @import("tensor").Tensor;

pub fn testEql() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const host: Location = .{ .host = alloc };
    const ctx = try Context.default();

    var left = try Tensor(.i32).fromSlice(
        gpu,
        &[_]i32{
            1, 2,
        },
        .{2},
    );
    defer left.deinit();
    var right = try Tensor(.i32).fromSlice(
        gpu,
        &[_]i32{
            1, 2,
            1, 2,
            1, 2,
            1, 2,
            1, 2,
            1, 2,
            1, 2,
        },
        .{ 7, 2 },
    );
    defer right.deinit();

    var eql_tensor = try eql(
        ctx,
        // TODO: try left.view(.{ -1, 2 }),
        // try left.view(.{2}),
        try left.view(.{2}),
        // try right.view(.{ -1, 2 }),
        right,
    );
    defer eql_tensor.deinit();

    const eql_h = try eql_tensor.move(host);
    defer eql_h.deinit();
    std.debug.print("eql_out = {any}\n", .{eql_h.s()});
}

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
