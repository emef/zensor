const std = @import("std");

const DType = @import("core").DType;

const ctypes = @import("coretypes.zig");
const mem = @import("mem.zig");

extern "c" fn cudaSetDevice(device: c_int) c_int;

pub const Device = union(enum) {
    default,
    id: i32,

    pub fn set(self: Device) ctypes.Error!void {
        const code = cudaSetDevice(self.resolve());
        if (code != 0) {
            return error.TODO;
        }
    }

    pub fn eql(self: Device, other: Device) bool {
        return self.resolve() == other.resolve();
    }

    fn resolve(self: Device) i32 {
        switch (self) {
            .default => return 0,
            .id => |x| return x,
        }
    }
};

pub fn device(id: i32) Device {
    return .{ .id = id };
}

pub fn DeviceMem(dtype_: DType) type {
    return struct {
        const Self = @This();

        pub const dtype = dtype_;
        pub const T = dtype_.toHostType();
        pub const elem_size = @sizeOf(T);

        // device memory ptr
        ptr: ?*anyopaque,
        // number of elements
        len: usize,
        // cuda device
        device: Device,

        pub fn init(dev: Device, n: usize) ctypes.Error!Self {
            var ptr: ?*anyopaque = null;
            try dev.set();
            const code = mem.cudaMalloc(&ptr, elem_size * n);
            if (code != 0) {
                return error.TODO;
            }

            return Self{
                .ptr = ptr,
                .len = n,
                .device = dev,
            };
        }

        pub fn deinit(self: Self) void {
            if (mem.cudaFree(self.ptr) != 0) {
                @panic("failed to free cuda memory");
            }
        }

        pub fn ptrOffset(self: Self, offset: usize) *anyopaque {
            const offset_bytes = elem_size * offset;
            return @ptrFromInt(@intFromPtr(self.ptr.?) + offset_bytes);
        }

        pub fn hostToDevice(self: Self, src: []const T) ctypes.Error!void {
            if (src.len != self.len) {
                return error.TODO;
            }

            const n_bytes = elem_size * src.len;

            const code = mem.cudaMemcpy(
                self.ptr,
                src.ptr,
                n_bytes,
                mem.cudaMemcpyHostToDevice,
            );

            if (code != 0) {
                return error.TODO;
            }
        }

        pub fn deviceToHost(self: Self, range: ctypes.ElemRange, dest: []T) ctypes.Error!void {
            if (range.len != dest.len) {
                return error.WrongSize;
            }

            const offset_ptr = self.ptrOffset(range.offset);
            const n_bytes = elem_size * range.len;

            const code = mem.cudaMemcpy(
                dest.ptr,
                offset_ptr,
                n_bytes,
                mem.cudaMemcpyDeviceToHost,
            );

            if (code != 0) {
                return error.TODO;
            }
        }

        pub fn deviceToDevice(
            self: Self,
            src_range: ctypes.ElemRange,
            src: DeviceMem(dtype),
        ) ctypes.Error!void {
            if (src_range.len != self.len) {
                return error.WrongSize;
            }

            const offset_ptr = src.ptrOffset(src_range.offset);
            const n_bytes = elem_size * src_range.len;

            const code = mem.cudaMemcpy(
                src.ptr,
                offset_ptr,
                n_bytes,
                mem.cudaMemcpyDeviceToDevice,
            );

            if (code != 0) {
                return error.TODO;
            }
        }
    };
}
