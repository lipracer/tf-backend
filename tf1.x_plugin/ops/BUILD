# load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "backend_ops",
    srcs = ["backend_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/compiler/jit/tf1.x_plugin:backend_include",
    ],
)
