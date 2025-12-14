const std = @import("std");

const cuda = @import("cuda");
const cublas = cuda.cublas;
const DType = @import("core").DType;
const SmallVec = @import("core").SmallVec;

const Context = @import("context.zig");
const expectTensorEqual = @import("testing.zig").expectEqual;
const ops = @import("ops/root.zig");
const TensorIterator = @import("iterator.zig").TensorIterator;

pub const max_dims = 8;

pub const HostError = error{
    OutOfMemory,
};

pub const Error = HostError || cuda.Error || cublas.Error || error{
    WrongSize,
    WrongShape,
    WrongDevice,
    WrongType,
    NotImplemented,
    NotContiguous,
};

pub const Shape = SmallVec(i32, max_dims);
pub const Strides = SmallVec(i32, max_dims);

fn stridesFromShape(shape: Shape) Strides {
    var strides = Strides{
        .data = undefined,
        .len = shape.len,
    };

    if (strides.len == 0) {
        return strides;
    }

    strides.data[strides.len - 1] = 1;
    for (1..shape.len) |i| {
        const rev_i = shape.len - i - 1;
        strides.data[rev_i] = shape.data[rev_i + 1] * strides.data[rev_i + 1];
    }

    return strides;
}

pub fn HostMem(dtype: DType) type {
    return struct {
        const Self = @This();

        pub const T = dtype.toHostType();

        slice: []T,
        alloc: std.mem.Allocator,

        pub fn init(alloc: std.mem.Allocator, shape: Shape) HostError!Self {
            return Self{
                .slice = try alloc.alloc(T, shape.elems()),
                .alloc = alloc,
            };
        }

        pub fn deinit(self: Self) void {
            self.alloc.free(self.slice);
        }
    };
}

pub const Location = union(enum) {
    host: std.mem.Allocator,
    cuda: cuda.Device,

    pub fn eql(self: Location, other: Location) bool {
        return switch (self) {
            .host => other == .host,
            .cuda => |d1| switch (other) {
                .host => false,
                .cuda => |d2| d1.eql(d2),
            },
        };
    }
};

pub fn Storage(dtype: DType) type {
    return union(enum) {
        const Self = @This();

        host: HostMem(dtype),
        cuda: cuda.DeviceMem(dtype),

        pub fn elems(self: *Self) usize {
            switch (self) {
                .host => |h| h.slice.len,
                .device => |d| d.len,
            }
        }
    };
}

pub fn Tensor(dtype_: DType) type {
    return struct {
        const Self = @This();

        pub const dtype = dtype_;
        pub const T = dtype.toHostType();

        storage: Storage(dtype),
        shape: Shape,
        strides: Strides,
        offset: usize,

        // true if the storage (host or device) is owned by this
        // tensor. if it is not owned, then deinit() will not
        // free the memory in this tensor's storage.
        owned: bool,

        // set to true when this tensor's data has been moved.
        // it is invalid to use this tensor after it has been moved.
        moved: bool = false,

        pub fn deinit(self: Self) void {
            if (!self.owned or self.moved) {
                return;
            }

            switch (self.storage) {
                .host => |h| h.deinit(),
                .cuda => |d| d.deinit(),
            }
        }

        pub fn init(dest: Location, shapeOrDims: anytype) Error!Self {
            return switch (dest) {
                .host => |a| try Self.allocHost(a, shapeOrDims),
                .cuda => |device| try Self.allocDevice(device, shapeOrDims),
            };
        }

        pub fn format(self: Self, w: anytype) !void {
            _ = self;
            _ = w;
            return error.TODO;
        }

        pub fn allocDevice(device: cuda.Device, shapeOrDims: anytype) cuda.Error!Self {
            const shape = Shape.init(shapeOrDims);

            return Self{
                .offset = 0,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .storage = Storage(dtype){
                    .cuda = try cuda.DeviceMem(dtype).init(device, shape.elems()),
                },
                .owned = true,
            };
        }

        pub fn allocHost(alloc: std.mem.Allocator, shapeOrDims: anytype) HostError!Self {
            const shape = Shape.init(shapeOrDims);
            const host_mem = try HostMem(dtype).init(alloc, shape);

            return Self{
                .offset = 0,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .storage = Storage(dtype){
                    .host = host_mem,
                },
                .owned = true,
            };
        }

        pub fn fromSlice(
            dest: Location,
            mem: []const T,
            shapeOrDims: anytype,
        ) Error!Self {
            // TODO: support -1 dims and infer them
            const shape = Shape.init(shapeOrDims);
            switch (dest) {
                .host => |alloc| {
                    const out = try Self.allocHost(alloc, shape);
                    @memcpy(out.storage.host.slice, mem);
                    return out;
                },
                .cuda => |device| {
                    const out = try Self.allocDevice(device, shape);
                    try out.storage.cuda.hostToDevice(mem);
                    return out;
                },
            }
        }

        pub fn fromCudaBuffer(mem: cuda.DeviceMem(dtype), shapeOrDims: anytype) Error!Self {
            const shape = Shape.init(shapeOrDims);

            if (shape.elems() != mem.len) {
                return error.WrongSize;
            }

            return Self{
                .offset = 0,
                .shape = shape,
                .strides = stridesFromShape(shape),
                .storage = Storage(dtype){
                    .cuda = mem,
                },
                .owned = false,
            };
        }

        // NOTE: this is a convenience reference that is only
        // valid for host tensors. only use this if you are
        // sure this has host storage
        pub fn s(self: Self) []T {
            switch (self.storage) {
                .host => |host_mem| {
                    const elems = self.shape.elems();
                    return host_mem.slice[self.offset .. self.offset + elems];
                },
                .cuda => {
                    @panic("cannot use .s() on cuda tensor");
                },
            }
        }

        pub fn iter(self: Self) Error!TensorIterator(dtype) {
            return try TensorIterator(dtype).init(self);
        }

        pub fn move(self: *Self, dest: Location) !Self {
            if (self.moved) {
                return error.TODO;
            }

            const range = cuda.ElemRange{
                .offset = self.offset,
                .len = self.shape.elems(),
            };

            switch (dest) {
                .host => |alloc| {
                    switch (self.storage) {
                        .host => {
                            const movedTo = self.*;
                            self.moved = true;
                            return movedTo;
                        },
                        .cuda => |storageMem| {
                            const movedTo = try Self.allocHost(alloc, self.shape);
                            try storageMem.deviceToHost(range, movedTo.storage.host.slice);
                            self.deinit();
                            self.moved = true;
                            return movedTo;
                        },
                    }
                },
                .cuda => |destDevice| {
                    switch (self.storage) {
                        .cuda => |storageMem| {
                            if (!storageMem.device.eql(destDevice)) {
                                return error.NotImplemented;
                            }

                            const movedTo = self.*;
                            self.moved = true;
                            return movedTo;
                        },
                        .host => |hostMem| {
                            const movedTo = try Self.allocDevice(destDevice, self.shape);
                            try movedTo.storage.cuda.hostToDevice(
                                hostMem.slice[range.offset .. range.offset + range.len],
                            );
                            self.deinit();
                            self.moved = true;
                            return movedTo;
                        },
                    }
                },
            }
        }

        pub fn copy(self: Self, dest: Location) !Self {
            if (self.moved) {
                return error.TODO;
            }

            if (!self.isContiguous()) {
                return error.TODO;
            }

            const range = cuda.ElemRange{
                .offset = self.offset,
                .len = self.shape.elems(),
            };

            var copied: Self = undefined;
            switch (dest) {
                .host => |alloc| {
                    copied = try Self.allocHost(alloc, self.shape);
                    errdefer copied.deinit();

                    switch (self.storage) {
                        .cuda => |d| try d.deviceToHost(range, copied.storage.host.slice),
                        .host => |h| @memcpy(
                            copied.storage.host.slice,
                            h.slice[range.offset .. range.offset + range.len],
                        ),
                    }
                },
                .cuda => |device| {
                    copied = try Self.allocDevice(device, self.shape);
                    errdefer copied.deinit();

                    switch (self.storage) {
                        .host => |h| try copied.storage.cuda.hostToDevice(
                            h.slice[range.offset .. range.offset + range.len],
                        ),
                        .cuda => |d| try copied.storage.cuda.deviceToDevice(range, d),
                    }
                },
            }

            return copied;
        }

        pub fn loc(self: Self) Location {
            return switch (self.storage) {
                .host => |h| .{ .host = h.alloc },
                .cuda => |d| .{ .cuda = d.device },
            };
        }

        pub fn view(self: Self, shapeOrDims: anytype) !Self {
            var newShape = Shape.init(shapeOrDims);

            var newElems: usize = 1;
            var infer: ?usize = null;
            for (0..newShape.len) |i| {
                const dim = newShape.data[i];
                if (dim == -1) {
                    if (infer != null) {
                        return error.TODO;
                    }
                    infer = i;
                    continue;
                }

                newElems *= @intCast(dim);
            }

            if (infer) |i| {
                const total = self.shape.elems();
                if (@mod(total, newElems) != 0) {
                    return error.TODO;
                }

                newShape.data[i] = @intCast(total / newElems);
            }

            if (self.shape.elems() != newShape.elems()) {
                return error.WrongShape;
            }

            if (!self.isContiguous()) {
                return error.TODO;
            }

            var viewed = self;
            viewed.owned = false;
            viewed.shape = newShape;
            viewed.strides = stridesFromShape(newShape);
            return viewed;
        }

        pub fn expand(self: Self, shapeOrDims: anytype) !Self {
            // broadcast to an expanded shape, dimensions of
            // size 1 can be arbitrary expanded by setting
            // stride to 0. -1 implies keep dimension the same.
            // singleton dimensions must remain the same

            // TODO: this only handles new dimensions but doesn't
            // handle expanding dimensions of size 1.

            var newShape = Shape.init(shapeOrDims);

            if (newShape.len < self.shape.len) {
                // new shape must have at least the same number
                // of dimensions
                return error.WrongSize;
            }

            const new_dims = newShape.len - self.shape.len;

            // verify singleton dimensions are unchanged
            for (0..self.shape.len) |i| {
                const j = new_dims + i;
                if (newShape.data[j] == -1) {
                    newShape.data[j] = self.shape.data[i];
                } else if (newShape.data[j] != self.shape.data[i]) {
                    // singleton dimensons must remain unchanged
                    return error.WrongSize;
                }
            }

            // create new strides and set the expanded dim
            // strides to 0 (broadcast).
            var newStrides = stridesFromShape(newShape);
            for (0..new_dims) |i| {
                newStrides.data[i] = 0;
            }

            var expanded = self;
            expanded.owned = false;
            expanded.shape = newShape;
            expanded.strides = newStrides;
            return expanded;
        }

        pub fn squeezeDim(self: Self, dim: i32) Self {
            if (self.shape.at(dim) != 1) {
                @panic("cannot non size 1 dimension");
            }

            var newShape: Shape = .{ .len = 0 };
            var newStrides: Strides = .{ .len = 0 };

            for (0..self.shape.len) |i| {
                if (i == dim) continue;
                newShape.data[newShape.len] = self.shape.data[i];
                newShape.len += 1;
                newStrides.data[newStrides.len] = self.strides.data[i];
                newStrides.len += 1;
            }

            var squeezed = self;
            squeezed.owned = false;
            squeezed.shape = newShape;
            squeezed.strides = newStrides;
            return squeezed;
        }

        pub fn unsqueeze(self: Self, dim_: i32) Self {
            var dim = dim_;
            if (dim < 0) {
                dim = dim + self.shape.len + 1;
            }

            var newShape: Shape = .{ .len = self.shape.len + 1 };
            var newStrides: Strides = .{ .len = self.strides.len + 1 };

            var count: usize = 0;
            for (0..self.shape.len) |i| {
                if (count == dim) {
                    newShape.data[count] = 1;
                    newStrides.data[count] = 0;
                    count += 1;
                }
                newShape.data[count] = self.shape.data[i];
                newStrides.data[count] = self.strides.data[i];
                count += 1;
            }

            var unsqueezed = self;
            unsqueezed.owned = false;
            unsqueezed.shape = newShape;
            unsqueezed.strides = newStrides;
            return unsqueezed;
        }

        pub fn select(self: Self, selector: anytype) !Self {
            if (self.moved) {
                return error.TODO;
            }

            var indexes: SmallVec(i32, max_dims) = undefined;
            switch (@typeInfo(@TypeOf(selector))) {
                .int, .comptime_int => {
                    indexes.data[0] = @intCast(selector);
                    indexes.len = 1;
                },
                .@"struct" => {
                    indexes = SmallVec(i32, max_dims).init(selector);
                },
                else => @compileError("only int or tuple allowed in select()"),
            }

            var newOffset = self.offset;
            var newShape = self.shape;
            var newStrides = self.strides;
            var popped: usize = 0;
            for (0..indexes.len) |i| {
                if (indexes.data[i] == -1) {
                    continue;
                }

                newOffset += @intCast(indexes.data[i] * self.strides.data[i]);
                newShape = newShape.pop(i - popped);
                newStrides = newStrides.pop(i - popped);
                popped += 1;
            }

            return Self{
                .storage = self.storage,
                .offset = newOffset,
                .shape = newShape,
                .strides = newStrides,
                .owned = false,
            };
        }

        pub fn isContiguous(self: Self) bool {
            if (self.shape.len == 0) return true;

            var expected: i32 = 1;
            var i: i32 = @intCast(self.shape.len - 1);
            while (i >= 0) : (i -= 1) {
                const size = self.shape.at(i);
                const stride = self.strides.at(i);
                if (size == 0) return false;
                if (size == 1) continue;
                if (stride != expected) return false;
                expected *= size;
            }

            return true;
        }

        pub fn item(self: Self) Error!T {
            if (self.shape.elems() != 1) {
                return error.WrongSize;
            }

            switch (self.storage) {
                .host => |mem| {
                    return mem.slice[self.offset];
                },
                .cuda => |mem| {
                    const range = cuda.ElemRange{
                        .offset = self.offset,
                        .len = 1,
                    };

                    var buf: [1]T = undefined;
                    try mem.deviceToHost(range, &buf);
                    return buf[0];
                },
            }
        }

        // Transpose the matrix dimensions (last 2) of the tensor.
        pub fn mT(self: Self) Self {
            const n_dims = self.shape.len;

            var transposed = self;
            transposed.owned = false;

            if (n_dims >= 2) {
                std.mem.swap(
                    i32,
                    &transposed.shape.data[n_dims - 2],
                    &transposed.shape.data[n_dims - 1],
                );
                std.mem.swap(
                    i32,
                    &transposed.strides.data[n_dims - 2],
                    &transposed.strides.data[n_dims - 1],
                );
            }

            return transposed;
        }

        pub fn arange(dest: Location, comptime start: i32, comptime end: i32) !Self {
            if (dtype == .bool) @compileError("bool is unsupported for arange");

            const caster = DType.caster(.i32, dtype);
            var buf: [end - start]T = undefined;

            const n: usize = @intCast(end - start);
            for (0..n) |i| {
                const value: i32 = start + @as(i32, @intCast(i));
                buf[i] = caster.cast(value);
            }

            return try fromSlice(dest, &buf, .{end - start});
        }

        pub fn eql(self: Self, ctx: Context, other: anytype) Error!Tensor(.bool) {
            return try ops.eql(ctx, self, other);
        }

        pub fn all(self: Self, ctx: Context) Error!bool {
            return try ops.all(ctx, self);
        }

        pub fn matmul(self: Self, ctx: Context, rhs: Self) Error!Self {
            return try ops.matmul(ctx, self, rhs, dtype);
        }

        pub fn add(self: Self, ctx: Context, rhs: Self) Error!Self {
            _ = self;
            _ = ctx;
            _ = rhs;
            return error.NotImplemented;
        }
    };
}

test "tensor alloc device" {
    const device = cuda.device(0);
    const t = try Tensor(.f32).allocDevice(device, .{ 1, 5, 10 });
    defer t.deinit();

    try std.testing.expect(t.storage == .cuda);
    try std.testing.expect((Location{ .cuda = device }).eql(t.loc()));

    try std.testing.expectEqual(50, t.storage.cuda.len);
    try std.testing.expect(t.owned);
    try std.testing.expect(!t.moved);
}

test "tensor alloc host" {
    const t = try Tensor(.f32).allocHost(std.testing.allocator, .{ 1, 5, 10 });
    defer t.deinit();

    try std.testing.expect(t.storage == .host);
    try std.testing.expect(t.loc() == .host);

    try std.testing.expectEqual(50, t.storage.host.slice.len);
    try std.testing.expect(t.owned);
    try std.testing.expect(!t.moved);
}

test "tensor init device" {
    const device = cuda.device(0);
    const mem = try cuda.DeviceMem(.u32).init(device, 50);
    defer mem.deinit();

    const t = try Tensor(.u32).fromCudaBuffer(mem, .{ 1, 5, 10 });
    defer t.deinit();

    try std.testing.expect(t.storage == .cuda);
    try std.testing.expect(t.loc().eql(.{ .cuda = device }));
    try std.testing.expectEqual(50, t.storage.cuda.len);

    try std.testing.expect(!t.owned);
    try std.testing.expect(!t.moved);
}

test "tensor move from host borrow" {
    var t = try Tensor(.u16).allocHost(std.testing.allocator, .{ 1, 5, 10 });
    defer t.deinit();

    var movedHost = try t.move(.{ .host = std.testing.allocator });
    defer movedHost.deinit();

    try std.testing.expect(t.moved);
    try std.testing.expect(!movedHost.moved);
    try std.testing.expect(movedHost.loc() == .host);
    try std.testing.expectEqual(50, movedHost.storage.host.slice.len);

    const device = cuda.device(0);
    const movedCuda = try movedHost.move(.{ .cuda = device });
    defer movedCuda.deinit();

    try std.testing.expect(movedHost.moved);
    try std.testing.expect(!movedCuda.moved);
    try std.testing.expect(movedCuda.loc().eql(.{ .cuda = device }));
    try std.testing.expectEqual(50, movedCuda.storage.cuda.len);
}

test "tensor move from host alloc" {
    const alloc = std.testing.allocator;
    var t = try Tensor(.u16).allocHost(alloc, .{ 1, 5, 10 });
    defer t.deinit();

    var movedHost = try t.move(.{ .host = alloc });
    defer movedHost.deinit();

    try std.testing.expect(t.moved);
    try std.testing.expect(!movedHost.moved);
    try std.testing.expect(movedHost.loc() == .host);
    try std.testing.expectEqual(50, movedHost.storage.host.slice.len);

    const device = cuda.device(0);
    const movedCuda = try movedHost.move(.{ .cuda = device });
    defer movedCuda.deinit();

    try std.testing.expect(movedHost.moved);
    try std.testing.expect(!movedCuda.moved);
    try std.testing.expect(movedCuda.loc().eql(.{ .cuda = device }));
    try std.testing.expectEqual(50, movedCuda.storage.cuda.len);
}

test "tensor move from device alloc" {
    const alloc = std.testing.allocator;
    var t = try Tensor(.u16).allocDevice(cuda.device(0), .{ 1, 5, 10 });
    defer t.deinit();

    var movedHost = try t.move(.{ .host = alloc });
    defer movedHost.deinit();

    try std.testing.expect(t.moved);
    try std.testing.expect(!movedHost.moved);
    try std.testing.expect(movedHost.loc() == .host);
    try std.testing.expectEqual(50, movedHost.storage.host.slice.len);

    const device = cuda.device(0);
    const movedCuda = try movedHost.move(.{ .cuda = device });
    defer movedCuda.deinit();

    try std.testing.expect(movedHost.moved);
    try std.testing.expect(!movedCuda.moved);
    try std.testing.expect(movedCuda.loc().eql(.{ .cuda = device }));
    try std.testing.expectEqual(50, movedCuda.storage.cuda.len);
}

test "tensor copy host to host" {
    const alloc = std.testing.allocator;
    var t = try Tensor(.u16).allocHost(alloc, .{ 1, 5, 10 });
    defer t.deinit();

    var copy = try t.copy(.{ .host = alloc });
    defer copy.deinit();

    try std.testing.expect(!t.moved);
    try std.testing.expect(!copy.moved);
    try std.testing.expect(copy.loc() == .host);
    try std.testing.expectEqual(50, copy.storage.host.slice.len);
}

test "tensor copy host to device" {
    const alloc = std.testing.allocator;
    var t = try Tensor(.u16).allocHost(alloc, .{ 1, 5, 10 });
    defer t.deinit();

    const device = cuda.device(0);
    var copy = try t.copy(.{ .cuda = device });
    defer copy.deinit();

    try std.testing.expect(!t.moved);
    try std.testing.expect(!copy.moved);
    try std.testing.expect(copy.loc().eql(.{ .cuda = device }));
    try std.testing.expectEqual(50, copy.storage.cuda.len);
}

test "tensor copy device to host" {
    const device = cuda.device(0);
    var t = try Tensor(.u16).allocDevice(device, .{ 1, 5, 10 });
    defer t.deinit();

    const alloc = std.testing.allocator;
    var copy = try t.copy(.{ .host = alloc });
    defer copy.deinit();

    try std.testing.expect(!t.moved);
    try std.testing.expect(!copy.moved);
    try std.testing.expect(copy.loc() == .host);
    try std.testing.expectEqual(50, copy.storage.host.slice.len);
}

test "tensor copy device to device" {
    const device = cuda.device(0);
    var t = try Tensor(.u16).allocDevice(device, .{ 1, 5, 10 });
    defer t.deinit();

    var copy = try t.copy(.{ .cuda = device });
    defer copy.deinit();

    try std.testing.expect(!t.moved);
    try std.testing.expect(!copy.moved);
    try std.testing.expect(copy.loc().eql(t.loc()));
    try std.testing.expectEqual(50, copy.storage.cuda.len);
}

test "select" {
    const device = cuda.device(0);
    // const host: Location = .{ .host = std.testing.allocator };
    const gpu: Location = .{ .cuda = device };

    const x = try Tensor(.f32).fromSlice(
        gpu,
        &[_]f32{
            1.0,  2.0,
            3.0,  4.0,
            5.0,  6.0,

            7.0,  8.0,
            9.0,  10.0,
            11.0, 12.0,

            13.0, 14.0,
            15.0, 16.0,
            17.0, 18.0,
        },
        .{ 3, 3, 2 },
    );
    defer x.deinit();

    // TODO: should we refcount views? that would require deinit()
    // on everything.. but it would allow something like
    //   x = allocDevice(..)
    //   x = x.reshape(.{-1, 5})
    //   x = ...

    var selected = try x.select(.{ 2, 1 });
    try std.testing.expectEqual(1, selected.shape.len);
    try std.testing.expectEqual(2, selected.shape.at(0));
    try std.testing.expect(selected.shape.eql(.{2}));
    try std.testing.expect(selected.isContiguous());

    const subselected = try selected.select(0);
    try std.testing.expect(subselected.shape.eql(.{}));
    try std.testing.expect(subselected.isContiguous());
    try std.testing.expectEqual(15, try subselected.item());

    selected = try x.select(.{ 0, 1, 1 });
    try std.testing.expectEqual(0, selected.shape.len);
    try std.testing.expect(selected.isContiguous());
    try std.testing.expect(selected.shape.eql(.{}));
    try std.testing.expectEqual(4.0, try selected.item());
}

test "expand" {
    const host: Location = .{ .host = std.testing.allocator };
    const gpu: Location = .{ .cuda = cuda.device(0) };

    for (&[_]Location{ host, gpu }) |loc| {
        const x = try Tensor(.f32).fromSlice(
            loc,
            &[_]f32{
                1.0, 2.0, 3.0,
            },
            .{3},
        );
        defer x.deinit();

        try expectTensorEqual(
            try x.expand(.{ 2, 3 }),
            &[_]f32{
                1, 2, 3,
                1, 2, 3,
            },
        );
    }
}

test "unsqueeze" {
    const host: Location = .{ .host = std.testing.allocator };
    const x = try Tensor(.i32).fromSlice(host, &[16]i32{
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8,
    }, .{ 2, 2, 4 });
    defer x.deinit();

    const test_cases: []const struct {
        dim: i32,
        expect_err: bool = false,
    } = &.{
        .{ .dim = 0 },
        .{ .dim = 1 },
        .{ .dim = 2 },
        .{ .dim = -1 },
        .{ .dim = -2 },
        .{ .dim = -3 },
    };

    var out_buf: [16]i32 = undefined;

    for (test_cases) |tc| {
        const unsqueezed = x.unsqueeze(tc.dim);

        var it = try unsqueezed.iter();
        try it.into(&out_buf);
        try std.testing.expectEqualSlices(i32, x.s(), &out_buf);
    }
}

test "squeezeDim" {
    const host: Location = .{ .host = std.testing.allocator };
    const x = try Tensor(.i32).fromSlice(host, &[16]i32{
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8,
    }, .{ 1, 2, 2, 1, 4 });
    defer x.deinit();

    var squeezed = x.squeezeDim(0);
    try std.testing.expect(squeezed.shape.eql(.{ 2, 2, 1, 4 }));

    squeezed = squeezed.squeezeDim(2);
    try std.testing.expect(squeezed.shape.eql(.{ 2, 2, 4 }));
}

test "arange" {
    const device = cuda.device(0);
    const host: Location = .{ .host = std.testing.allocator };
    const gpu: Location = .{ .cuda = device };

    for (&[_]Location{ host, gpu }) |loc| {
        {
            var x = try Tensor(.i32).arange(loc, 0, 10);
            defer x.deinit();

            var x_out = try x.move(host);
            defer x_out.deinit();

            for (x_out.s(), 0..) |val, i| {
                try std.testing.expectEqual(@as(i32, @intCast(i)), val);
            }
        }

        {
            var x = try Tensor(.i32).arange(loc, -20, -10);
            defer x.deinit();

            var x_out = try x.move(host);
            defer x_out.deinit();

            for (x_out.s(), 0..) |val, i| {
                const expect: i32 = -20 + @as(i32, @intCast(i));
                try std.testing.expectEqual(expect, val);
            }
        }

        {
            var x = try Tensor(.i32).arange(loc, 10, 20);
            defer x.deinit();

            var x_out = try x.move(host);
            defer x_out.deinit();

            for (x_out.s(), 0..) |val, i| {
                const expect: i32 = 10 + @as(i32, @intCast(i));
                try std.testing.expectEqual(expect, val);
            }
        }
    }
}

test "mT" {
    const host: Location = .{ .host = std.testing.allocator };
    const x = try Tensor(.i32).fromSlice(host, &[16]i32{
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8,
    }, .{ 2, 2, 4 });
    defer x.deinit();

    const x_mt = x.mT();
    try std.testing.expect(x_mt.shape.eql(.{ 2, 4, 2 }));

    const expect = [16]i32{
        1, 5,
        2, 6,
        3, 7,
        4, 8,

        1, 5,
        2, 6,
        3, 7,
        4, 8,
    };

    var out: [16]i32 = undefined;
    var it = try x_mt.iter();
    try it.into(&out);

    try std.testing.expectEqualSlices(i32, &expect, &out);
}
