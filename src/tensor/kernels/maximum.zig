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
const KElementwise = @import("elementwise.zig").KElementwise;

pub fn KMaximum(dtype: DType) type {
    return KElementwise(
        dtype,
        "maximumContiguous_",
        "maximumStrided_",
    );
}

test KMaximum {
    const device = cuda.device(0);
    const gpu: Location = .{ .cuda = device };
    const stream = try cuda.Stream.init(device);
    defer stream.deinit();

    // contiguous
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
                20, 1,  -1, 3,  4,
                5,  -1, 7,  18, -4,
            },
            .{10},
        );
        defer b.deinit();

        const out = try Tensor(.i32).empty(gpu, a.shape);

        try cuda.launchKernel(
            device,
            try KMaximum(.i32).init(.{
                .a = a,
                .b = b,
                .out = out,
            }),
            stream,
        );

        try stream.sync();

        try expectTensorEqual(out, &[_]i32{
            20, 1, 2, 3,  4,
            5,  6, 7, 18, 9,
        });
    }

    // strided
    {
        var a = try Tensor(.i32).fromSlice(
            gpu,
            &[_]i32{
                0,  -1, 2,  -3, 4,
                -5, 6,  -7, 8,  -9,
            },
            .{ 1, 2, 5 },
        );
        defer a.deinit();

        var b = try Tensor(.i32).fromSlice(
            gpu,
            &[_]i32{0},
            .{1},
        );
        defer b.deinit();

        const b_bcast = try b.expand(a.shape);

        const out = try Tensor(.i32).empty(gpu, a.shape);

        try cuda.launchKernel(
            device,
            try KMaximum(.i32).init(.{
                .a = a,
                .b = b_bcast,
                .out = out,
            }),
            stream,
        );

        try stream.sync();

        try expectTensorEqual(out, &[_]i32{
            0, 0, 2, 0, 4,
            0, 6, 0, 8, 0,
        });
    }
}
