const std = @import("std");

const DType = @import("core").DType;

const Location = @import("root.zig").Location;
const Shape = @import("root.zig").Shape;
const Tensor = @import("root.zig").Tensor;

pub const Descriptor = struct {
    dtype: DType,
    shape: Shape,
    ptr: []u8,
};

pub const DescriptorMap = std.StringArrayHashMap(Descriptor);

const HeaderJSONEntry = struct {
    dtype: []const u8,
    shape: []u32,
    data_offsets: []u32,
};

const HeaderJSON = std.json.ArrayHashMap(HeaderJSONEntry);

const dtype_map = std.StaticStringMap(DType).initComptime([_]struct { []const u8, DType }{
    .{ "BOOL", .bool },
    .{ "F16", .f16 },
    .{ "F32", .f32 },
    .{ "F64", .f64 },
    .{ "I8", .i8 },
    .{ "I16", .i16 },
    .{ "I32", .i32 },
    .{ "I64", .i64 },
    .{ "U8", .u8 },
    .{ "U16", .u16 },
    .{ "U32", .u32 },
    .{ "U64", .u64 },
});

pub const SafeTensors = struct {
    ptr: ?[]align(std.heap.page_size_min) u8,
    arena: std.heap.ArenaAllocator,
    descriptors: DescriptorMap,

    pub fn deinit(self: SafeTensors) void {
        self.arena.deinit();

        if (self.ptr) |ptr| {
            std.posix.munmap(ptr);
        }
    }

    pub fn mmap(alloc: std.mem.Allocator, path: []const u8) !SafeTensors {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const file_size = stat.size;

        const ptr = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        const buf = ptr[0..file_size];

        const header_len = std.mem.readInt(u64, buf[0..8], .little);
        const header_buf = buf[8 .. 8 + header_len];

        const parsed = try std.json.parseFromSlice(HeaderJSON, alloc, header_buf, .{});
        defer parsed.deinit();

        var arena = std.heap.ArenaAllocator.init(alloc);
        var descs = DescriptorMap.init(arena.allocator());

        const data_buf = buf[8 + header_len ..];

        const header: HeaderJSON = parsed.value;
        var it = header.map.iterator();
        while (it.next()) |entry| {
            const name = try arena.allocator().dupe(u8, entry.key_ptr.*);
            const entry_val = entry.value_ptr.*;

            var shape: Shape = .{ .len = 0 };
            for (entry_val.shape) |size| {
                shape.data[shape.len] = @intCast(size);
                shape.len += 1;
            }

            const dtype = dtype_map.get(entry_val.dtype) orelse unreachable;

            std.debug.print("safetensor: {s} {any}\n", .{ name, entry_val });

            if (entry_val.data_offsets.len != 2) unreachable;
            const start = entry_val.data_offsets[0];
            const end = entry_val.data_offsets[1];

            std.debug.print("ptr slice {d}..{d} (file size {d})\n", .{ start, end, file_size });

            try descs.put(name, .{
                .dtype = dtype,
                .shape = shape,
                .ptr = data_buf[start..end],
            });
        }

        return SafeTensors{
            .arena = arena,
            .descriptors = descs,
            .ptr = ptr,
        };
    }

    pub fn get(
        self: SafeTensors,
        comptime dtype: DType,
        loc: Location,
        name: []const u8,
    ) !Tensor(dtype) {
        const desc = self.descriptors.get(name) orelse return error.KeyNotFound;
        if (desc.dtype != dtype) return error.WrongType;

        const T = dtype.toHostType();
        const ptr: []T = @ptrCast(@alignCast(desc.ptr));

        switch (loc) {
            .host => |alloc| {
                // memoryView creates a host tensors w/o copying the data
                return Tensor(dtype).memoryView(alloc, ptr, desc.shape);
            },
            .cuda => {
                return try Tensor(dtype).fromSlice(loc, ptr, desc.shape);
            },
        }
    }
};

test "SafeTensors.mmap" {
    const alloc = std.testing.allocator;
    const host: Location = .{ .host = alloc };
    const tensor_path = "src/tensor/fixtures/test.safetensors";
    const tensors = try SafeTensors.mmap(alloc, tensor_path);
    defer tensors.deinit();

    var it = tensors.descriptors.iterator();
    while (it.next()) |entry| {
        std.debug.print("loaded tensor descriptor from mmap: {s}\n", .{entry.key_ptr.*});
    }

    const weight = try tensors.get(.f32, host, "layers.0.weight");
    defer weight.deinit();

    std.debug.print("weight {f}: {any}\n", .{ weight.shape, weight.s()[0..10] });
}
