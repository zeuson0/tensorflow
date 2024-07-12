load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility_macros",
    hdrs = ["compatibility_macros.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

# LINT.IfChange

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":compatibility_macros",
        ":cppmath",
    ],
)

# LINT.ThenChange(//tensorflow/lite/kernels/internal/BUILD)
