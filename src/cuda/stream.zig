const core = @import("coretypes.zig");
const device = @import("device.zig");

extern "c" fn cudaStreamCreate(pStream: *core.cudaStream_t) c_int;
extern "c" fn cudaStreamDestroy(stream: core.cudaStream_t) c_int;
extern "c" fn cudaStreamSynchronize(stream: core.cudaStream_t) c_int;

pub const Stream = struct {
    ptr: core.cudaStream_t,

    pub fn init(dev: device.Device) core.Error!Stream {
        try dev.set();

        var ptr: core.cudaStream_t = undefined;
        if (cudaStreamCreate(&ptr) != 0) {
            return error.TODO;
        }

        return Stream{ .ptr = ptr };
    }

    pub fn sync(self: Stream) core.Error!void {
        if (cudaStreamSynchronize(self.ptr) != 0) {
            return error.TODO;
        }
    }

    pub fn deinit(self: Stream) void {
        if (cudaStreamDestroy(self.ptr) != 0) {
            @panic("failed to destroy cuda stream");
        }
    }
};
