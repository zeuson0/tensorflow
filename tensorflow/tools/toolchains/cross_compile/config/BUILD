package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = {
        "container-image": "docker://us.gcr.io/tensorflow-testing/srnitin-macos-cross-compile@sha256:a7bf437b3adf87ebcfa510c84780f68cafb6c31f1af8f95acc0bc7ea37cedb31",
        "OSFamily": "Linux",
    },
)

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

platform(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

toolchain(
    name = "macos-x86-cross-compile-cc-toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    toolchain = "//tensorflow/tools/toolchains/cross_compile/cc:macos_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

toolchain(
    name = "macos-x86-cross-compile-py-toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    toolchain = "@python_x86_64-unknown-linux-gnu//:python_runtimes",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
