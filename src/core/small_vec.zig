const std = @import("std");

pub fn SmallVec(T: type, max_dims: usize) type {
    return struct {
        const Self = @This();

        data: [max_dims]T = undefined,
        len: u4,

        pub fn init(vecOrDims: anytype) Self {
            if (@TypeOf(vecOrDims) == Self) {
                return vecOrDims;
            }

            var vec = Self{ .data = undefined, .len = 0 };
            const info = @typeInfo(@TypeOf(vecOrDims));
            const fields = info.@"struct".fields;
            if (fields.len > max_dims) {
                @panic("too many dimensions!");
            }

            inline for (fields, 0..) |field, i| {
                vec.data[i] = @intCast(@field(vecOrDims, field.name));
                vec.len += 1;
            }
            return vec;
        }

        pub fn format(
            self: @This(),
            writer: anytype,
        ) !void {
            if (self.len == 0) {
                try writer.print("()\n", .{});
                return;
            }

            try writer.print("(", .{});
            for (0..self.len - 1) |i| {
                try writer.print("{d}, ", .{self.data[i]});
            }
            try writer.print("{d})", .{self.data[self.len - 1]});
        }

        pub fn at(self: Self, i: i32) T {
            return self.data[self.index(i)];
        }

        pub fn replace(self: Self, i: i32, val: T) Self {
            var replaced = self;
            replaced.data[self.index(i)] = val;
            return replaced;
        }

        fn index(self: Self, i: anytype) u4 {
            if (i >= self.len or i < -@as(i32, @intCast(self.len))) {
                @panic("out of bounds");
            }

            if (i < 0) {
                return self.len - @as(u4, @intCast(@abs(i)));
            }

            return @intCast(i);
        }

        pub fn pop(self: Self, pop_idx_: anytype) Self {
            const pop_idx = self.index(pop_idx_);
            var popped = Self{
                .data = undefined,
                .len = 0,
            };

            for (0..self.len) |i| {
                if (i == pop_idx) continue;
                popped.data[popped.len] = self.data[i];
                popped.len += 1;
            }

            return popped;
        }

        pub fn eql(self: Self, otherShapeOrDims: anytype) bool {
            const other = Self.init(otherShapeOrDims);
            if (self.len != other.len) {
                return false;
            }

            return std.mem.eql(T, self.data[0..self.len], other.data[0..self.len]);
        }

        pub fn elems(self: Self) usize {
            var count: usize = 1;
            for (0..self.len) |d| {
                if (d < 0) {
                    @panic("cannot take elem count with negative dims");
                }
                count *= @intCast(self.data[d]);
            }
            return count;
        }
    };
}
