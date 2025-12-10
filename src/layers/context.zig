const cuda = @import("cuda");

pub const Context = struct {
    stream: cuda.Stream,
};
