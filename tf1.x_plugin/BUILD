
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "tf_plugin.so",
    srcs = [
        "tf_plugin.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/compiler/jit/tf1.x_plugin/ops:backend_ops",
        "//tensorflow/compiler/jit/tf1.x_plugin/kernels:backend_ops",
        "//tensorflow/compiler/jit/tf1.x_plugin/optimization_pass:plugin_passes",
        "//third_party/eigen3",
        ":backend_include",
        "@com_google_protobuf//:protobuf",
    ],
    linkshared = 1,
    # copts = [],
    linkopts = [],
)

cc_library(
    name = "backend_include",
    hdrs = glob(["include/*.h", "include/**/*.h"]),
)
