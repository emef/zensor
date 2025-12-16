const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core = b.addModule("core", .{
        .root_source_file = b.path("src/core/root.zig"),
        .target = target,
    });

    const cuda = b.addModule("cuda", .{
        .root_source_file = b.path("src/cuda/root.zig"),
        .target = target,
    });
    cuda.addImport("core", core);

    const tensor = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor/root.zig"),
        .target = target,
    });

    tensor.addImport("core", core);
    tensor.addImport("cuda", cuda);
    tensor.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
    tensor.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
    tensor.linkSystemLibrary("cudart", .{});
    tensor.linkSystemLibrary("cublas", .{});
    addCudaSources(b, tensor, "src/tensor/kernels/");

    const nn = b.addModule("nn", .{
        .root_source_file = b.path("src/nn/root.zig"),
        .target = target,
    });
    nn.addImport("core", core);
    nn.addImport("tensor", tensor);
    nn.addImport("cuda", cuda);

    const mod = b.addModule("zensor", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    mod.addImport("core", core);
    mod.addImport("cuda", cuda);
    mod.addImport("tensor", tensor);
    mod.addImport("nn", nn);

    const examples_exe = b.addExecutable(.{
        .name = "zensor_examples",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/examples/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zensor", .module = mod },
            },
        }),
    });

    examples_exe.linkLibC();

    b.installArtifact(examples_exe);

    const run_step = b.step("run", "Run the examples");
    const run_cmd = b.addRunArtifact(examples_exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const tensor_tests = b.addTest(.{
        .root_module = tensor,
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);
    tensor_tests.linkLibC();

    const nn_tests = b.addTest(.{
        .root_module = nn,
    });
    const run_nn_tests = b.addRunArtifact(nn_tests);
    nn_tests.linkLibC();

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_tensor_tests.step);
    test_step.dependOn(&run_nn_tests.step);

    b.installArtifact(tensor_tests);
}

fn addCudaSources(
    b: *std.Build,
    mod: *std.Build.Module,
    comptime src_root: []const u8,
) void {
    var cuda_dir = std.fs.cwd().openDir(src_root, .{ .iterate = true }) catch {
        std.debug.print("kernel dir not found", .{});
        unreachable;
    };
    defer cuda_dir.close();

    var walker = cuda_dir.iterate();
    while (walker.next() catch null) |entry| {
        if (entry.kind == .file) {
            if (std.mem.endsWith(u8, entry.name, ".cu")) {
                const nvcc_cmd = b.addSystemCommand(&.{
                    "nvcc",
                    "-G",
                    "-g",
                    "-allow-unsupported-compiler",
                    "-c",
                });
                const full_path = b.pathJoin(&.{ src_root, entry.name });
                nvcc_cmd.addFileArg(b.path(full_path));
                nvcc_cmd.addArg("-o");

                var obj_name: [std.fs.max_path_bytes]u8 = undefined;
                _ = std.mem.replace(u8, entry.name, ".cu", ".o", &obj_name);
                const cuda_obj = nvcc_cmd.addOutputFileArg(obj_name[0 .. entry.name.len - 1]);
                mod.addObjectFile(cuda_obj);
            }
        }
    }
}
