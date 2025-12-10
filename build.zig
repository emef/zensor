const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.
    //

    const core = b.addModule("core", .{
        .root_source_file = b.path("src/core/root.zig"),
        .target = target,
    });

    const cuda = b.addModule("cuda", .{
        .root_source_file = b.path("src/cuda/root.zig"),
        .target = target,
    });
    cuda.addImport("core", core);
    cuda.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
    cuda.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
    addCudaSources(b, cuda, "src/kernels/");
    cuda.linkSystemLibrary("cudart", .{});
    cuda.linkSystemLibrary("cublas", .{});

    const tensor = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor/root.zig"),
        .target = target,
    });

    tensor.addImport("core", core);
    tensor.addImport("cuda", cuda);

    const kernels = b.addModule("kernels", .{
        .root_source_file = b.path("src/kernels/kernels.zig"),
        .target = target,
    });

    kernels.addImport("core", core);
    kernels.addImport("cuda", cuda);
    kernels.addImport("tensor", tensor);

    const layers = b.addModule("layers", .{
        .root_source_file = b.path("src/layers/layers.zig"),
        .target = target,
    });
    layers.addImport("core", core);
    layers.addImport("tensor", tensor);
    layers.addImport("cuda", cuda);

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("zensor", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    mod.addImport("core", core);
    mod.addImport("kernels", kernels);
    mod.addImport("cuda", cuda);
    mod.addImport("tensor", tensor);

    const exe = b.addExecutable(.{
        .name = "zensor",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "zensor" is the name you will use in your source code to
                // import this module (e.g. `@import("zensor")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "zensor", .module = mod },
            },
        }),
    });

    exe.linkLibC();

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const kernels_tests = b.addTest(.{
        .root_module = kernels,
    });
    const run_kernels_tests = b.addRunArtifact(kernels_tests);
    kernels_tests.linkLibC();

    const tensor_tests = b.addTest(.{
        .root_module = tensor,
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);
    tensor_tests.linkLibC();

    const layers_tests = b.addTest(.{
        .root_module = layers,
    });
    const run_layers_tests = b.addRunArtifact(layers_tests);
    layers_tests.linkLibC();

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_kernels_tests.step);
    test_step.dependOn(&run_tensor_tests.step);
    test_step.dependOn(&run_layers_tests.step);
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
