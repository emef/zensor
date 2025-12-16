const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const tensor = @import("../root.zig");
const Tensor = tensor.Tensor;
const common = @import("common.zig");
const TensorInfo = common.TensorInfo;
const KernelArgs = common.KernelArgs;

pub fn KElementwise(
    dtype: DType,
    k_prefix_contiguous: []const u8,
    k_prefix_strided: []const u8,
) type {
    const k_fn_name_contiguous = k_prefix_contiguous ++ @tagName(dtype);
    const k_fn_name_strided = k_prefix_strided ++ @tagName(dtype);
    const k_fn_contiguous = @extern(*const anyopaque, .{ .name = k_fn_name_contiguous });
    const k_fn_strided = @extern(*const anyopaque, .{ .name = k_fn_name_strided });

    return struct {
        const Self = @This();

        dims: cuda.KernelDims,

        // kernel args:
        a_ptr: ?*anyopaque,
        a_info: TensorInfo,
        b_ptr: ?*anyopaque,
        b_info: TensorInfo,
        out_ptr: ?*anyopaque,
        contiguous: bool,
        kernel_args: KernelArgs,

        pub const Args = struct {
            a: Tensor(dtype),
            b: Tensor(dtype),
            out: Tensor(dtype),
            threads_per_block: u32 = 256,
        };

        fn checkCompatible(t: anytype, out_elems: usize) !void {
            if (t.shape.elems() != out_elems) {
                return error.WrongSize;
            }

            if (t.storage != .cuda) {
                return error.WrongDevice;
            }
        }

        pub fn init(args: Args) !Self {
            const threads: u32 = @intCast(args.threads_per_block);
            const out_elems: usize = @intCast(args.out.shape.elems());
            const tile_size: u32 = @divFloor(@as(u32, @intCast(out_elems)) + threads - 1, threads);

            try checkCompatible(args.a, out_elems);
            try checkCompatible(args.b, out_elems);
            try checkCompatible(args.out, out_elems);

            if (!args.out.isContiguous()) {
                return error.NotContiguous;
            }

            const contiguous = args.a.isContiguous() and args.b.isContiguous();

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
                .a_info = TensorInfo{
                    .strides = @bitCast(args.a.strides.data),
                    .ndim = @intCast(args.a.strides.len),
                },
                .b_ptr = args.b.storage.cuda.ptrOffset(args.b.offset),
                .b_info = TensorInfo{
                    .strides = @bitCast(args.b.strides.data),
                    .ndim = @intCast(args.b.strides.len),
                },
                .out_ptr = args.out.storage.cuda.ptrOffset(args.out.offset),
                .contiguous = contiguous,
                .kernel_args = KernelArgs{
                    .shape = @bitCast(args.out.shape.data),
                    .ndim = @intCast(args.out.shape.len),
                    .elems = @intCast(out_elems),
                },
            };
        }

        pub fn spec(self: *Self) cuda.KernelSpec {
            var args: cuda.KernelArgs = undefined;
            var func: *const anyopaque = undefined;

            switch (self.contiguous) {
                true => {
                    args = cuda.kernelArgs(&.{
                        @ptrCast(&self.a_ptr),
                        @ptrCast(&self.b_ptr),
                        @ptrCast(&self.out_ptr),
                        @ptrCast(&self.kernel_args.elems),
                    });

                    func = k_fn_contiguous;
                },
                false => {
                    args = cuda.kernelArgs(&.{
                        @ptrCast(&self.a_ptr),
                        @ptrCast(&self.a_info),
                        @ptrCast(&self.b_ptr),
                        @ptrCast(&self.b_info),
                        @ptrCast(&self.out_ptr),
                        @ptrCast(&self.kernel_args),
                    });

                    func = k_fn_strided;
                },
            }

            return cuda.KernelSpec{
                .func = func,
                .dims = self.dims,
                .args = args,
            };
        }
    };
}
