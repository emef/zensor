const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;
const Location = @import("tensor").Location;
const Tensor = @import("tensor").Tensor;

extern const eqlTyped_i32: anyopaque;

pub fn KEql(dtype: DType) type {
    return struct {
        const Self = @This();

        dims: cuda.KernelDims,

        // kernel args:
        a_ptr: ?*anyopaque,
        b_ptr: ?*anyopaque,
        out_ptr: ?*anyopaque,
        elems: i32,

        pub const Args = struct {
            a: Tensor(dtype),
            b: Tensor(dtype),
            out: Tensor(.bool),
            threads_per_block: u32 = 256,
        };

        fn checkCompatible(t: anytype, out_elems: usize) !void {
            if (t.shape.elems() != out_elems) {
                return error.WrongSize;
            }

            if (t.storage != .cuda) {
                return error.WrongDevice;
            }

            if (!t.isContiguous()) {
                return error.NotContiguous;
            }
        }

        pub fn init(args: Args) !Self {
            const threads: u32 = @intCast(args.threads_per_block);
            const out_elems: usize = @intCast(args.out.shape.elems());
            const tile_size: u32 = @divFloor(@as(u32, @intCast(out_elems)) + threads - 1, threads);

            try checkCompatible(args.a, out_elems);
            try checkCompatible(args.b, out_elems);
            try checkCompatible(args.out, out_elems);

            return Self{
                .dims = cuda.KernelDims{
                    .grid = cuda.Dim3{
                        .x = tile_size,
                        .y = 1,
                        .z = 1,
                    },
                    .block = cuda.Dim3{
                        .x = threads,
                        .y = 1,
                        .z = 1,
                    },
                },
                .a_ptr = args.a.storage.cuda.ptrOffset(args.a.offset),
                .b_ptr = args.a.storage.cuda.ptrOffset(args.b.offset),
                .out_ptr = args.a.storage.cuda.ptrOffset(args.out.offset),
                .elems = @intCast(out_elems),
            };
        }

        pub fn spec(self: *Self) cuda.KernelSpec {
            const args = cuda.kernelArgs(&.{
                @ptrCast(&self.a_ptr),
                @ptrCast(&self.b_ptr),
                @ptrCast(&self.out_ptr),
                @ptrCast(&self.elems),
            });

            return cuda.KernelSpec{
                .func = &eqlTyped_i32,
                .dims = self.dims,
                .args = args,
            };
        }
    };
}

test KEql {
    const alloc = std.testing.allocator;
    const host: Location = .{ .host = alloc };
    const device = cuda.device(0);
    const cuda_loc: Location = .{ .cuda = device };
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    var x1 = try Tensor(.i32).fromSlice(
        cuda_loc,
        &[_]i32{
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
        },
        .{ 1, 2, 5 },
    );
    defer x1.deinit();

    var x2 = try Tensor(.i32).fromSlice(
        cuda_loc,
        &[_]i32{
            0, 1,  -1, 3, 4,
            5, -1, 7,  8, -1,
        },
        .{10},
    );
    defer x2.deinit();

    var x3 = try Tensor(.i32).fromSlice(
        cuda_loc,
        &[_]i32{
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
        },
        .{ 2, 5 },
    );
    defer x3.deinit();

    var d_out = try Tensor(.bool).allocDevice(device, .{10});
    defer d_out.deinit();

    try cuda.launchKernel(
        device,
        try KEql(.i32).init(.{
            .a = x1,
            .b = x2,
            .out = d_out,
        }),
        stream,
    );

    const h_out = try d_out.move(host);
    defer h_out.deinit();
}
