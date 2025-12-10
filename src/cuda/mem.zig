pub extern "c" fn cudaMalloc(devPtr: *?*anyopaque, size: usize) c_int;
pub extern "c" fn cudaFree(devPtr: ?*anyopaque) c_int;
pub extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) c_int;
pub extern "c" fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) c_int;

pub const cudaMemcpyHostToHost: c_int = 0;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;
pub const cudaMemcpyDeviceToDevice: c_int = 3;
