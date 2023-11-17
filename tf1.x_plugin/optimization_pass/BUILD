
cc_library(
    name = "plugin_passes",
    hdrs = [
        "white_list.h"
    ],
    srcs = [
        "build_backend_ops_pass.cc",
        "white_list.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:encapsulate_util",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/jit:xla_activity_listener",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",

        "//tensorflow/compiler/jit/tf1.x_plugin/kernels:backend_ops",
        "//tensorflow/compiler/jit/tf1.x_plugin:backend_include",

        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
