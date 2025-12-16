const DType = @import("core").DType;
const tensor = @import("tensor");
const Tensor = tensor.Tensor;
const Context = tensor.Context;
const Error = tensor.Error;
const Location = tensor.Location;

pub fn ReLU(dtype: DType) type {
    return struct {
        const Self = @This();
        const T = dtype.toHostType();

        zero: Tensor(dtype),

        pub fn init(loc: Location) Error!Self {
            return Self{
                .zero = try Tensor(dtype).fromSlice(loc, &[_]T{0}, .{1}),
            };
        }

        pub fn apply(self: Self, ctx: Context, x: Tensor(dtype)) Error!Tensor(dtype) {
            return try x.maximum(ctx, self.zero);
        }
    };
}
