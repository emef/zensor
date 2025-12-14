const std = @import("std");

const DType = @import("core").DType;
const SmallVec = @import("core").SmallVec;

const tensor_ = @import("tensor.zig");
const Tensor = tensor_.Tensor;
const Location = tensor_.Location;
const max_dims = tensor_.max_dims;
const Shape = tensor_.Shape;
const Strides = tensor_.Strides;
const Error = tensor_.Error;

pub fn TensorIterator(dtype: DType) type {
    const T: type = dtype.toHostType();

    return struct {
        const Self = @This();

        shape: SmallVec(usize, max_dims),
        strides: SmallVec(usize, max_dims),
        slice: []T,
        indices: [max_dims]usize = @splat(0),
        done: bool = false,

        pub fn init(tensor: anytype) Error!Self {
            const tensor_dtype: DType = @TypeOf(tensor).dtype;
            if (tensor_dtype != dtype) {
                @compileError("wrong iterator type for tensor");
            }

            const slice: []T = switch (tensor.storage) {
                .host => |h| h.slice,
                else => return error.WrongDevice,
            };

            var shape: SmallVec(usize, max_dims) = .{
                .data = undefined,
                .len = tensor.shape.len,
            };
            var strides: SmallVec(usize, max_dims) = .{
                .data = undefined,
                .len = tensor.strides.len,
            };

            for (0..tensor.shape.len) |i| {
                shape.data[i] = @intCast(tensor.shape.data[i]);
            }

            for (0..tensor.strides.len) |i| {
                strides.data[i] = @intCast(tensor.strides.data[i]);
            }

            return Self{
                .shape = shape,
                .strides = strides,
                .slice = slice,
                .done = slice.len == 0,
            };
        }

        pub fn next(self: *Self) ?T {
            if (self.done) return null;

            var offset: usize = 0;
            for (0..self.shape.len) |d| {
                offset += self.indices[d] * self.strides.data[d];
            }

            var d: usize = self.shape.len;
            while (d > 0) {
                d -= 1;
                self.indices[d] += 1;
                if (self.indices[d] < self.shape.data[d]) break;
                self.indices[d] = 0;
                if (d == 0) self.done = true;
            }

            return self.slice[offset];
        }
    };
}

test TensorIterator {
    const host: Location = .{ .host = std.testing.allocator };

    var data = [_]i32{
        0,  1,  2,  3,
        4,  5,  6,  7,

        8,  9,  10, 11,
        12, 13, 14, 15,
    };

    const tensor = try Tensor(.i32).fromSlice(host, &data, .{ 2, 2, 4 });
    defer tensor.deinit();

    var it = try tensor.iter();
    var i: i32 = 0;
    while (it.next()) |el| : (i += 1) {
        try std.testing.expectEqual(@as(i32, @intCast(i)), el);
    }

    // works with views
    const viewed = try tensor.view(.{ 2, 8 });
    i = 0;
    it = try viewed.iter();
    while (it.next()) |el| : (i += 1) {
        try std.testing.expectEqual(@as(i32, @intCast(i)), el);
    }

    // workes with broadcast
    const expanded = try viewed.expand(.{ 3, -1, -1 });
    it = try expanded.iter();
    for (0..3) |_| {
        i = 0;
        for (0..16) |expect| {
            const el = it.next() orelse return error.TestFailed;
            try std.testing.expectEqual(@as(i32, @intCast(expect)), el);
        }
    }
}
