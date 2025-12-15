const std = @import("std");

const cuda = @import("cuda");
const DType = @import("core").DType;

const tensor = @import("../root.zig");
const Location = tensor.Location;
const Tensor = tensor.Tensor;
const expectTensorEqual = tensor.testing.expectEqual;
const common = @import("common.zig");
const TensorInfo = common.TensorInfo;
const KernelArgs = common.KernelArgs;

extern const addStrided_bool: anyopaque;
extern const addStrided_f16: anyopaque;
extern const addStrided_f32: anyopaque;
extern const addStrided_f64: anyopaque;
extern const addStrided_i8: anyopaque;
extern const addStrided_i16: anyopaque;
extern const addStrided_i32: anyopaque;
extern const addStrided_i64: anyopaque;
extern const addStrided_u8: anyopaque;
extern const addStrided_u16: anyopaque;
extern const addStrided_u32: anyopaque;
extern const addStrided_u64: anyopaque;

extern const addContiguous_bool: anyopaque;
extern const addContiguous_f16: anyopaque;
extern const addContiguous_f32: anyopaque;
extern const addContiguous_f64: anyopaque;
extern const addContiguous_i8: anyopaque;
extern const addContiguous_i16: anyopaque;
extern const addContiguous_i32: anyopaque;
extern const addContiguous_i64: anyopaque;
extern const addContiguous_u8: anyopaque;
extern const addContiguous_u16: anyopaque;
extern const addContiguous_u32: anyopaque;
extern const addContiguous_u64: anyopaque;

pub fn KAdd(dtype: DType) type {
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

                    func = switch (dtype) {
                        .bool => &addContiguous_bool,
                        .f16 => &addContiguous_f16,
                        .f32 => &addContiguous_f32,
                        .f64 => &addContiguous_f64,
                        .i8 => &addContiguous_i8,
                        .i16 => &addContiguous_i16,
                        .i32 => &addContiguous_i32,
                        .i64 => &addContiguous_i64,
                        .u8 => &addContiguous_u8,
                        .u16 => &addContiguous_u16,
                        .u32 => &addContiguous_u32,
                        .u64 => &addContiguous_u64,
                    };
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

                    func = switch (dtype) {
                        .bool => &addStrided_bool,
                        .f16 => &addStrided_f16,
                        .f32 => &addStrided_f32,
                        .f64 => &addStrided_f64,
                        .i8 => &addStrided_i8,
                        .i16 => &addStrided_i16,
                        .i32 => &addStrided_i32,
                        .i64 => &addStrided_i64,
                        .u8 => &addStrided_u8,
                        .u16 => &addStrided_u16,
                        .u32 => &addStrided_u32,
                        .u64 => &addStrided_u64,
                    };
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

test KAdd {
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    {
        var a = try Tensor(.i32).fromSlice(
            gpu,
            &[_]i32{
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
            },
            .{ 1, 2, 5 },
        );
        defer a.deinit();

        var b = try Tensor(.i32).fromSlice(
            gpu,
            &[_]i32{
                0, 1,  -1, 3,  4,
                5, -1, 7,  18, -4,
            },
            .{10},
        );
        defer b.deinit();

        const out = try Tensor(.i32).init(gpu, a.shape);

        try cuda.launchKernel(
            device,
            try KAdd(.i32).init(.{
                .a = a,
                .b = b,
                .out = out,
            }),
            stream,
        );

        try stream.sync();

        try expectTensorEqual(out, &[_]i32{
            0,  2, 1,  6,  8,
            10, 5, 14, 26, 5,
        });
    }
}
