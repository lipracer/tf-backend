cc_library(
    name = "backend_ops",
    srcs = ["backend_ops.cc"],
    hdrs = [
        "backend_ops.h", 
        "adaptor.hpp"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:xla_activity_listener",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:tf_allocator_adapter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "//tensorflow/compiler/jit/tf1.x_plugin:backend_include",
    ],
    alwayslink = 1,
    # linkstatic = 1,
)
