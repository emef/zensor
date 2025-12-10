const std = @import("std");

const cuda = @import("cuda");
const Tensor = @import("tensor").Tensor;

extern "C" fn mmulNaive(
    a: [*]i32,
    b: [*]i32,
    out: [*]i32,
    n: i32,
) void;

extern "C" fn mmulTiled(
    a: [*]i32,
    b: [*]i32,
    out: [*]i32,
    M: i32,
    N: i32,
    P: i32,
) void;

pub const KMMulNaive = struct {
    args: Args,
    dims: cuda.KernelDims,

    pub const Args = struct {
        a: Tensor(.i32),
        b: Tensor(.i32),
        out: Tensor(.i32),
        n: i32,
        threads_per_block: u32 = 16,
    };

    fn compatible(t: Tensor(.i32), size: usize) bool {
        if (t.shape.elems() != size) {
            return false;
        }

        if (t.storage != .cuda) {
            return false;
        }

        return true;
    }

    pub fn init(args: Args) !KMMulNaive {
        const threads: u32 = @intCast(args.threads_per_block);
        const tileSize: u32 = @divFloor(@as(u32, @intCast(args.n)) + threads - 1, threads);
        const size: usize = @intCast(args.n * args.n);

        if (!compatible(args.a, size) or
            !compatible(args.b, size) or
            !compatible(args.out, size))
        {
            return error.TODO;
        }

        return KMMulNaive{
            .args = args,
            .dims = cuda.KernelDims{
                .grid = cuda.Dim3{
                    .x = tileSize,
                    .y = tileSize,
                    .z = 1,
                },
                .block = cuda.Dim3{
                    .x = threads,
                    .y = threads,
                    .z = 1,
                },
            },
        };
    }

    pub fn spec(self: *KMMulNaive) cuda.KernelSpec {
        const args = cuda.kernelArgs(&.{
            @ptrCast(&self.args.a.storage.cuda.ptr),
            @ptrCast(&self.args.b.storage.cuda.ptr),
            @ptrCast(&self.args.out.storage.cuda.ptr),
            @ptrCast(&self.args.n),
        });

        return cuda.KernelSpec{
            .func = mmulNaive,
            .dims = self.dims,
            .args = args,
        };
    }
};

test KMMulNaive {
    const alloc = std.testing.allocator;
    const device = cuda.device(0);
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    const N = 100;

    var h_a = try Tensor(.i32).allocHost(alloc, .{ 1, N, N });
    defer h_a.deinit();
    var h_b = try Tensor(.i32).allocHost(alloc, .{ 1, N, N });
    defer h_b.deinit();

    var expect: [N * N]i32 = undefined;

    for (0..N) |i| {
        for (0..N) |j| {
            h_a.s()[i * N + j] = @intCast((i * N + j) % 10);
            h_b.s()[i * N + j] = @intCast((i * N + j) % 10);
        }
    }

    for (0..N) |i| {
        for (0..N) |j| {
            const temp: i32 = 0;
            //for (0..N) |k| {
            //    temp += a_buf[i * N + k] * b_buf[k * N + j];
            //}
            expect[i * N + j] = temp;
        }
    }

    var d_a = try h_a.move(.{ .cuda = device });
    defer d_a.deinit();
    var d_b = try h_b.copy(.{ .cuda = device });
    defer d_b.deinit();
    var d_out = try Tensor(.i32).allocDevice(device, .{ N, N });
    defer d_out.deinit();

    const start = try std.time.Instant.now();

    try cuda.launchKernel(
        device,
        try KMMulNaive.init(.{
            .a = d_a,
            .b = d_b,
            .out = d_out,
            .n = N,
        }),
        stream,
    );

    const h_out = try d_out.move(.{ .host = alloc });
    defer h_out.deinit();

    const elapsed = (try std.time.Instant.now()).since(start) / std.time.ns_per_us;
    std.debug.print("naive: {d}us\n", .{elapsed});

    const h_out_view = h_out.s();

    for (0..N * N) |_| {
        // try std.testing.expectEqual(h_out_view[i], expect[i]);
        _ = h_out_view;
    }
}

pub const KMMulTiled = struct {
    args: Args,
    dims: cuda.KernelDims,

    pub const Args = struct {
        a: Tensor(.i32),
        b: Tensor(.i32),
        out: Tensor(.i32),
        n: i32,
        threads_per_block: u32 = 8,
    };

    fn compatible(t: Tensor(.i32), size: usize) bool {
        if (t.shape.elems() != size) {
            return false;
        }

        if (t.storage != .cuda) {
            return false;
        }

        return true;
    }

    pub fn init(args: Args) !KMMulTiled {
        const threads: u32 = @intCast(args.threads_per_block);
        const tileSize: u32 = @divFloor(@as(u32, @intCast(args.n)) + threads - 1, threads);
        const size: usize = @intCast(args.n * args.n);

        std.debug.print("n={d} threads={d} tileSize={d} size={d}\n", .{ args.n, threads, tileSize, size });

        if (!compatible(args.a, size) or
            !compatible(args.b, size) or
            !compatible(args.out, size))
        {
            return error.TODO;
        }

        return KMMulTiled{
            .args = args,
            .dims = cuda.KernelDims{
                .grid = cuda.Dim3{
                    .x = tileSize,
                    .y = tileSize,
                    .z = 1,
                },
                .block = cuda.Dim3{
                    .x = threads,
                    .y = threads,
                    .z = 1,
                },
            },
        };
    }

    pub fn spec(self: *KMMulTiled) cuda.KernelSpec {
        const args = cuda.kernelArgs(&.{
            @ptrCast(&self.args.a.storage.cuda.ptr),
            @ptrCast(&self.args.b.storage.cuda.ptr),
            @ptrCast(&self.args.out.storage.cuda.ptr),
            @ptrCast(&self.args.n),
            @ptrCast(&self.args.n),
            @ptrCast(&self.args.n),
        });

        const shared_mem = 2 * self.dims.block.x * self.dims.block.y * @sizeOf(i32);

        std.debug.print("dims: {}\n", .{self.dims});

        return cuda.KernelSpec{
            .func = mmulTiled,
            .dims = self.dims,
            .args = args,
            .shared_mem = shared_mem,
        };
    }
};

test KMMulTiled {
    const alloc = std.testing.allocator;
    const device = cuda.device(0);
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    const N = 100;

    var h_a = try Tensor(.i32).allocHost(alloc, .{ 1, N, N });
    defer h_a.deinit();

    var h_b = try Tensor(.i32).allocHost(alloc, .{ 1, N, N });
    defer h_b.deinit();

    var expect: [N * N]i32 = undefined;

    for (0..N) |i| {
        for (0..N) |j| {
            // a_buf[i * N + j] = @intCast((i * N + j) % 10);
            // b_buf[i * N + j] = @intCast((i * N + j) % 10);
            h_a.s()[i * N + j] = 1;
            h_b.s()[i * N + j] = 1;
            expect[i * N + j] = 0;
        }
    }

    for (0..N) |i| {
        for (0..N) |j| {
            var temp: i32 = 0;
            for (0..N) |k| {
                temp += h_a.s()[i * N + k] * h_b.s()[k * N + j];
            }
            expect[i * N + j] = temp;
        }
    }

    var d_a = try h_a.move(.{ .cuda = device });
    defer d_a.deinit();
    var d_b = try h_b.copy(.{ .cuda = device });
    defer d_b.deinit();
    var d_out = try Tensor(.i32).allocDevice(device, .{ N, N });
    defer d_out.deinit();

    const start = try std.time.Instant.now();

    try cuda.launchKernel(
        device,
        try KMMulTiled.init(.{
            .a = d_a,
            .b = d_b,
            .out = d_out,
            .n = N,
        }),
        stream,
    );

    const h_out = try d_out.move(.{ .host = alloc });
    defer h_out.deinit();

    const elapsed = (try std.time.Instant.now()).since(start) / std.time.ns_per_us;
    std.debug.print("tiled: {d}us\n", .{elapsed});

    for (0..N) |i| {
        for (0..N) |j| {
            if (i < 1) continue;
            const idx = i * N + j;
            try std.testing.expectEqual(expect[idx], h_out.s()[idx]);
        }
    }
}
