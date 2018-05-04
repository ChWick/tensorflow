#include "tensorflow/core/user_ops/fuzzy_ctc_loss_calculator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FuzzyCTCLoss")
    .Input("inputs: float")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Attr("ignore_longer_outputs_than_inputs: bool = false")
    .Output("loss: float")
    .Output("gradient: float")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle inputs;
        ShapeHandle labels_indices;
        ShapeHandle labels_values;
        ShapeHandle sequence_length;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &labels_indices));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &labels_values));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sequence_length));

        DimensionHandle unused;
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(labels_indices, 0),
                    c->Dim(labels_values, 0), &unused));

        // Get batch size from inputs and sequence_length, and update inputs
        // with the merged batch_size since it is returned.
        DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(
                c->Merge(c->Dim(inputs, 0), c->Dim(sequence_length, 0), &batch_size));
        TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 0, batch_size, &inputs));

        c->set_output(0, c->Vector(batch_size));
        c->set_output(1, inputs);
        return Status::OK();
});

REGISTER_OP("FuzzyCTCGreedyDecoder")
    .Input("inputs: float")
    .Input("sequence_length: int32")
    .Output("decoded_indices: int64")
    .Output("decoded_values: int64")
    .Output("decoded_shape: int64")
    .Output("log_probability: float")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle inputs;
        ShapeHandle sequence_length;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

        // Get batch size from inputs and sequence_length.
        DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(
                c->Merge(c->Dim(inputs, 0), c->Dim(sequence_length, 0), &batch_size));

        DimensionHandle total_decoded_outputs = c->UnknownDim();
        c->set_output(0, c->Matrix(total_decoded_outputs, 2));
        c->set_output(1, c->Vector(total_decoded_outputs));
        c->set_output(2, c->Vector(2));
        c->set_output(3, c->Matrix(batch_size, 1));
        return Status::OK();
});

}  // namespace tensorflow
