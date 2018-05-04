#define EIGEN_USE_THREADS

#include <limits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/user_ops/fuzzy_ctc_decoder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

inline float RowMax(const TTypes<float>::UnalignedConstMatrix& m, int r,
                    int* c, int omit = -1) {
  *c = 0;
  CHECK_LT(0, m.dimension(1));
  float p = m(r, 0);
  if (omit == 0) {
      *c = 1;
      p = m(r, 1);
  }
  for (int i = 1; i < m.dimension(1); ++i) {
    if (i == omit) { continue; } 
    if (m(r, i) > p) {
      p = m(r, i);
      *c = i;
    }
  }
  return p;
}

class FuzzyCTCDecodeHelper {
 public:
  FuzzyCTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  Status ValidateInputsGenerateOutputs(
      OpKernelContext* ctx, const Tensor** inputs, const Tensor** seq_len,
      Tensor** log_prob, OpOutputList* decoded_indices,
      OpOutputList* decoded_values, OpOutputList* decoded_shape) const {
    Status status = ctx->input("inputs", inputs);
    if (!status.ok()) return status;
    status = ctx->input("sequence_length", seq_len);
    if (!status.ok()) return status;

    const TensorShape& inputs_shape = (*inputs)->shape();

    if (inputs_shape.dims() != 3) {
      return errors::InvalidArgument("inputs is not a 3-Tensor");
    }

    const int64 batch_size = inputs_shape.dim_size(0);
    const int64 max_time = inputs_shape.dim_size(1);

    if (max_time == 0) {
      return errors::InvalidArgument("max_time is 0");
    }
    if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
      return errors::InvalidArgument("sequence_length is not a vector");
    }

    if (!(batch_size == (*seq_len)->dim_size(0))) {
      return errors::FailedPrecondition(
          "len(sequence_length) != batch_size.  ",
          "len(sequence_length):  ", (*seq_len)->dim_size(0),
          " batch_size: ", batch_size);
    }

    auto seq_len_t = (*seq_len)->vec<int32>();

    for (int b = 0; b < batch_size; ++b) {
      if (!(seq_len_t(b) <= max_time)) {
        return errors::FailedPrecondition("sequence_length(", b,
                                          ") <= ", max_time);
      }
    }

    Status s = ctx->allocate_output(
        "log_probability", TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;

    s = ctx->output_list("decoded_indices", decoded_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", decoded_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", decoded_shape);
    if (!s.ok()) return s;

    return Status::OK();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int> > >& sequences,
      OpOutputList* decoded_indices, OpOutputList* decoded_values,
      OpOutputList* decoded_shape) const {
    // Calculate the total number of entries for each path
    const int64 batch_size = sequences.size();
    std::vector<int64> num_entries(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto& batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      Tensor* p_indices = nullptr;
      Tensor* p_values = nullptr;
      Tensor* p_shape = nullptr;

      const int64 p_num = num_entries[p];

      Status s =
          decoded_indices->allocate(p, TensorShape({p_num, 2}), &p_indices);
      if (!s.ok()) return s;
      s = decoded_values->allocate(p, TensorShape({p_num}), &p_values);
      if (!s.ok()) return s;
      s = decoded_shape->allocate(p, TensorShape({2}), &p_shape);
      if (!s.ok()) return s;

      auto indices_t = p_indices->matrix<int64>();
      auto values_t = p_values->vec<int64>();
      auto shape_t = p_shape->vec<int64>();

      int64 max_decoded = 0;
      int64 offset = 0;

      for (int64 b = 0; b < batch_size; ++b) {
        auto& p_batch = sequences[b][p];
        int64 num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        std::copy_n(p_batch.begin(), num_decoded, &values_t(offset));
        for (int64 t = 0; t < num_decoded; ++t, ++offset) {
          indices_t(offset, 0) = b;
          indices_t(offset, 1) = t;
        }
      }

      shape_t(0) = batch_size;
      shape_t(1) = max_decoded;
    }
    return Status::OK();
  }

 private:
  int top_paths_;
  TF_DISALLOW_COPY_AND_ASSIGN(FuzzyCTCDecodeHelper);
};

class FuzzyCTCGreedyDecoderOp : public OpKernel {
  typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor> >
      InputMap;
 public:
  explicit FuzzyCTCGreedyDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* seq_len;
    Tensor* log_prob = nullptr;
    OpOutputList decoded_indices;
    OpOutputList decoded_values;
    OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    const TensorShape& inputs_shape = inputs->shape();

    std::vector<InputMap> input_list_b;
    const int64 batch_size = inputs_shape.dim_size(0);
    const int64 max_time = inputs_shape.dim_size(1);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    auto inputs_t = inputs->tensor<float, 3>();

    for (std::size_t b = 0; b < batch_size; ++b) {
      input_list_b.emplace_back(inputs_t.data() + b * max_time * num_classes,
                                max_time, num_classes);
    }
    auto seq_len_t = seq_len->vec<int32>();
    auto log_prob_t = log_prob->matrix<float>();

    log_prob_t.setZero();

    // Assumption: the blank index is num_classes - 1
    int blank_index = num_classes - 1;

    // Perform best path decoding
    std::vector<std::vector<std::vector<int> > > sequences(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      const auto &inputs_b = input_list_b[b];
      sequences[b].resize(1);
      auto& sequence = sequences[b][0];
      int current_best_index = -1;
      float current_best_log_p = -100000000;
      for (int t = 0; t < seq_len_t(b); ++t) {
        auto p_blank = inputs_b(t, blank_index);
        if (p_blank > 0.7) {
            if (current_best_index >= 0) {
              if (current_best_index != blank_index) {
                sequence.push_back(current_best_index);
              }
              current_best_index = -1;
              current_best_log_p = 0;
            }
            log_prob_t(b, 0) *= p_blank;
        } else {
            int max_class_indices;
            //auto log_p = inputs_b.row(t).head(num_classes - 1).maxCoeff(&max_class_indices);
            auto log_p = inputs_b.row(t).maxCoeff(&max_class_indices);
            log_prob_t(b, 0) *= log_p;
            if (log_p > current_best_log_p) {
              current_best_index = max_class_indices;
              current_best_log_p = log_p;
            }
        }
      }
      if (current_best_index >= 0 && current_best_index != blank_index) {
        sequence.push_back(current_best_index);
      }
              
    }

    OP_REQUIRES_OK(
        ctx, decode_helper_.StoreAllDecodedSequences(
                 sequences, &decoded_indices, &decoded_values, &decoded_shape));
  }

 private:
  FuzzyCTCDecodeHelper decode_helper_;

  TF_DISALLOW_COPY_AND_ASSIGN(FuzzyCTCGreedyDecoderOp);
};

REGISTER_KERNEL_BUILDER(Name("FuzzyCTCGreedyDecoder").Device(DEVICE_CPU),
                        FuzzyCTCGreedyDecoderOp);

}  // end namespace tensorflow
