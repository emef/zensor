pub const cudaStream_t = ?*anyopaque;

pub const Dim3 = extern struct {
    x: c_uint,
    y: c_uint,
    z: c_uint,
};

pub const ElemRange = struct {
    offset: usize,
    len: usize,
};

pub const Error = error{
    TODO,
    WrongSize,
};

pub const cudaDataType_t = enum(c_int) {
    r_32f = 0,
    r_64f = 1,
    r_16f = 2,
    r_8i = 3,
    c_32f = 4,
    c_64f = 5,
    c_16f = 6,
    c_8i = 7,
    r_8u = 8,
    c_8u = 9,
    r_32i = 10,
    c_32i = 11,
    r_32u = 12,
    c_32u = 13,
    r_16bf = 14,
    c_16bf = 15,
    r_4i = 16,
    c_4i = 17,
    r_4u = 18,
    c_4u = 19,
    r_16i = 20,
    c_16i = 21,
    r_16u = 22,
    c_16u = 23,
    r_64i = 24,
    c_64i = 25,
    r_64u = 26,
    c_64u = 27,
    r_8f_e4m3 = 28,
    r_8f_e5m2 = 29,

    pub fn fromType(T: type) cudaDataType_t {
        return switch (T) {
            u64 => .r_64u,
            i64 => .r_64i,
            f64 => .r_64f,
            u32 => .r_32u,
            i32 => .r_32i,
            f32 => .r_32f,
            u16 => .r_16u,
            i16 => .r_16i,
            f16 => .r_16f,
            i8 => .r_8i,
            u8 => .r_8u,
            u4 => .r_4u,
            i4 => .r_4i,
            else => @compileError("no matching cuda dtype"),
        };
    }
};
