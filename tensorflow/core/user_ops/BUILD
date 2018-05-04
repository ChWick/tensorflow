load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "fuzzy_ctc_op.so",
    srcs = ["fuzzy_ctc_loss_calculator.cc",
            "fuzzy_ctc_loss_calculator.h",
            "fuzzy_ctc_loss_op.cc",
            "fuzzy_ctc_decoder.h",
            "fuzzy_ctc_decoder_op.cc",
            "fuzzy_ctc_ops.cc",
    ],
    deps = [
            "//tensorflow/core/util/ctc:ctc_loss_util_lib",
    ],
)

