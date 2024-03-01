licenses(["restricted"])

package(default_visibility = ["//visibility:public"])


alias(
    name = "numpy",
    actual = select({
        "@local_tsl//tsl:linux_to_macos_crosscompile": "@pypi_numpy_macos//:pkg",
        "//conditions:default": "@pypi_numpy//:pkg",
    }),
)

alias(
    name = "headers",
    actual = select({
        "@local_tsl//tsl:linux_to_macos_crosscompile": "@pypi_numpy_macos//:numpy_headers",
        "//conditions:default": "@pypi_numpy//:numpy_headers",
    }),
)

genrule(
    name = "dummy",
    outs = ["tf_numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
