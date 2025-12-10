pub const DType = enum {
    bool,
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f16,
    f32,
    f64,

    pub fn fromHostType(T: type) DType {
        return switch (T) {
            bool => .bool,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            else => @compileError("invalid tensor dtype"),
        };
    }

    pub fn toHostType(comptime self: DType) type {
        return switch (self) {
            .bool => bool,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .f16 => f16,
            .f32 => f32,
            .f64 => f64,
        };
    }

    pub fn promote(comptime lhs: DType, comptime rhs: DType) DType {
        if (lhs == .bool) return rhs;
        if (rhs == .bool) return lhs;

        const min_bits = @max(lhs.bits(), rhs.bits());
        const is_signed = lhs.isSigned() or rhs.isSigned();
        const is_float = lhs.isFloat() or rhs.isFloat();

        if (is_float) {
            if (min_bits >= 64) return .f64;
            if (min_bits >= 32) return .f32;
            return .f16;
        }

        if (is_signed) {
            if (min_bits >= 64) return .i64;
            if (min_bits >= 32) return .i32;
            if (min_bits >= 16) return .i16;
            return .i8;
        }

        if (min_bits >= 64) return .u64;
        if (min_bits >= 32) return .u32;
        if (min_bits >= 16) return .u16;
        return .u8;
    }

    pub fn caster(comptime src: DType, comptime dest: DType) type {
        const src_type = src.toHostType();
        const dest_type = dest.toHostType();

        if (src_type == dest_type) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return el;
                }
            };
        }

        if (src == .bool and dest.isFloat()) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @floatFromInt(@intFromBool(el));
                }
            };
        }

        if (src == .bool) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @intFromBool(el);
                }
            };
        }

        if (dest == .bool) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return el != 0;
                }
            };
        }

        if (src.isFloat() and dest.isFloat()) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @floatCast(el);
                }
            };
        }

        if (dest.isFloat()) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @floatFromInt(el);
                }
            };
        }

        if (src.isFloat()) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @intFromFloat(el);
                }
            };
        }

        if (!src.isFloat() and !dest.isFloat()) {
            return struct {
                pub inline fn cast(el: src_type) dest_type {
                    return @intCast(el);
                }
            };
        }

        @compileError("invalid type promotion");
    }

    fn bits(comptime self: DType) comptime_int {
        return switch (@typeInfo(toHostType(self))) {
            .bool => 1,
            .int => |int_info| int_info.bits,
            .float => |float_info| float_info.bits,
            else => @compileError("unhandled type"),
        };
    }

    fn isSigned(comptime self: DType) bool {
        return switch (@typeInfo(toHostType(self))) {
            .bool => false,
            .int => |int_info| int_info.signedness == .signed,
            .float => true,
            else => @compileError("unhandled type"),
        };
    }

    fn isFloat(comptime self: DType) bool {
        return @typeInfo(self.toHostType()) == .float;
    }

    pub fn size(comptime self: DType) comptime_int {
        return @sizeOf(self.toHostType());
    }
};
