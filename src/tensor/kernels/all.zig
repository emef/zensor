const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const tensor = @import("../root.zig");
const Location = tensor.Location;
const Tensor = tensor.Tensor;
const common = @import("common.zig");
const TensorInfo = common.TensorInfo;
const KernelArgs = common.KernelArgs;

extern const allStrided: anyopaque;
extern const allContiguous: anyopaque;

pub const KAll = struct {
    dims: cuda.KernelDims,
    a_ptr: ?*anyopaque,
    a_info: TensorInfo,
    contiguous: bool,
    out_ptr: ?*anyopaque,
    kernel_args: KernelArgs,

    pub const Args = struct {
        a: Tensor(.bool),
        out: Tensor(.i32),
        threads_per_block: u32 = 256,
    };

    pub fn init(args: Args) tensor.Error!KAll {
        const threads: u32 = @intCast(args.threads_per_block);
        const elems: usize = @intCast(args.a.shape.elems());
        const elems_u32: u32 = @intCast(elems);
        const grid_size = (elems_u32 + threads * 2 - 1) / (threads * 2);

        if (args.a.storage != .cuda or args.out.storage != .cuda) {
            return error.WrongDevice;
        }

        try args.out.storage.cuda.hostToDevice(&[_]i32{1});

        return KAll{
            .dims = cuda.KernelDims{
                .grid = cuda.Dim3{
                    .x = grid_size,
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
            .a_info = TensorInfo{
                .strides = @bitCast(args.a.strides.data),
                .ndim = @intCast(args.a.strides.len),
            },
            .contiguous = args.a.isContiguous(),
            .out_ptr = args.out.storage.cuda.ptrOffset(args.out.offset),
            .kernel_args = KernelArgs{
                .shape = @bitCast(args.a.shape.data),
                .ndim = @intCast(args.a.shape.len),
                .elems = @intCast(elems),
            },
        };
    }

    pub fn spec(self: *KAll) cuda.KernelSpec {
        if (self.contiguous) {
            const args = cuda.kernelArgs(&.{
                @ptrCast(&self.a_ptr),
                @ptrCast(&self.out_ptr),
                @ptrCast(&self.kernel_args.elems),
            });

            return cuda.KernelSpec{
                .func = &allContiguous,
                .dims = self.dims,
                .args = args,
            };
        } else {
            const args = cuda.kernelArgs(&.{
                @ptrCast(&self.a_ptr),
                @ptrCast(&self.a_info),
                @ptrCast(&self.out_ptr),
                @ptrCast(&self.kernel_args),
            });

            return cuda.KernelSpec{
                .func = &allStrided,
                .dims = self.dims,
                .args = args,
                .shared_mem = self.dims.block.x * @sizeOf(i32),
            };
        }
    }
};

test KAll {
    const alloc = std.testing.allocator;
    const host: Location = .{ .host = alloc };
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    const buf_true: [1024]bool = @splat(true);
    var buf_false: [1024]bool = @splat(true);
    buf_false[506] = false;

    const test_cases: []const struct {
        a: []const bool,
        expand: bool,
        expect: bool,
    } = &.{
        .{
            .a = &buf_true,
            .expand = false,
            .expect = true,
        },
        .{
            .a = &buf_true,
            .expand = true,
            .expect = true,
        },
        .{
            .a = &buf_false,
            .expand = false,
            .expect = false,
        },
        .{
            .a = &buf_false,
            .expand = true,
            .expect = false,
        },
    };

    const out = try Tensor(.i32).init(gpu, .{1});
    defer out.deinit();

    for (test_cases, 0..) |tc, i| {
        var a = try Tensor(.bool).fromSlice(gpu, tc.a, .{tc.a.len});
        defer a.deinit();

        if (tc.expand) {
            a = try a.expand(.{ 2, tc.a.len });
            try std.testing.expect(!a.isContiguous());
        }

        try cuda.launchKernel(
            device,
            try KAll.init(.{
                .a = a,
                .out = out,
            }),
            stream,
        );

        var checked = true;
        for (tc.a) |val| checked &= val;
        try std.testing.expectEqual(tc.expect, checked);

        try stream.sync();

        const h_out = try out.copy(host);
        defer h_out.deinit();

        const result = h_out.s()[0] != 0;
        std.debug.print("test {d}: expand={any} expect={any} got={d}\n", .{ i, tc.expand, tc.expect, h_out.s()[0] });
        try std.testing.expectEqual(tc.expect, result);
    }
}
