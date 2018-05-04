/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_

#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/ctc/ctc_loss_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace ctc {

class FuzzyCTCLossCalculator {
  // Connectionist Temporal Classification Loss
  //
  // Implementation by chwick@.
  //
  // The CTC Loss layer learns a *transition* probability value for each
  // input time step.  The transitions are on the class alphabet
  //   {0, 1, ..., N-2}
  // where N is the depth of the input layer (the size of the alphabet is N-1).
  // Note: The token N-1 is reserved for the "no transition" output, so
  // make sure that your input layer has a depth that's one larger than
  // the set of classes you're training on.  Also make sure that your
  // training labels do not have a class value of N-1, as training will skip
  // these examples.
  //
  // Reference materials:
  //  GravesTh: Alex Graves, "Supervised Sequence Labeling with Recurrent
  //    Neural Networks" (PhD Thesis), Technische Universit¨at M¨unchen.
 public:
  typedef std::vector<std::vector<int> > LabelSequences;
  typedef Eigen::MatrixXf Matrix;
  typedef Eigen::MatrixXd MatrixXd;
  typedef Eigen::VectorXf VectorXf;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::RowVectorXf RowVectorXf;
  typedef Eigen::RowVectorXd RowVectorXd;
  typedef std::vector<MatrixXd> TargetSequences;
  typedef Eigen::ArrayXf Array;
  typedef Eigen::MatrixXi LabelMatrix;
  typedef Eigen::Map<const Eigen::MatrixXf> InputMap;
  typedef Eigen::Map<Eigen::MatrixXf> OutputMap;

  FuzzyCTCLossCalculator(int blank_index)
      : blank_index_(blank_index), minimum_input_(1e-5), lower_bound_(1e-9)
      , skip_penalty_(-5.0) {}

  template <typename VectorIn, typename VectorOut, typename MatrixIn,
            typename MatrixOut>
  Status CalculateLoss(const VectorIn& seq_len, const LabelSequences& labels,
                       const std::vector<MatrixIn>& inputs,
                       bool ignore_longer_outputs_than_inputs,
                       VectorOut* loss,
                       std::vector<MatrixOut>* gradients,
                       DeviceBase::CpuWorkerThreads* workers = nullptr) const;

 private:
  void CalculateForwardVariables(const MatrixXd &log_match, MatrixXd *forward) const;
  void CalculateGradient(const Matrix &inputs, const Matrix &aligned, Matrix* dy) const;


  void CalculateForwardBackward(const MatrixXd& log_match, MatrixXd *alpha_beta) const;
  void AlignTargets(const Matrix& inputs, const MatrixXd& targets, Matrix *aligned) const;

  void MakeTarget(const std::vector<int>& l, int num_classes, MatrixXd *target) const;

  // Helper function that calculates the targets all
  // batches at the same time, and identifies errors for any given
  // batch.  Return value:
  //    max_{b in batch_size} targets[b].row()
  template <typename Vector>
  Status PopulateTargets(bool preprocess_collapse_repeated,
                         bool ignore_longer_outputs_than_inputs, int batch_size,
                         int num_classes, const Vector& seq_len,
                         const LabelSequences& labels, size_t* max_u_prime,
                         TargetSequences* targets) const;

  // Utility indices for the CTC algorithm.
  int blank_index_;

  const float minimum_input_;
  const float lower_bound_;
  const float skip_penalty_;
};

template <typename VectorIn, typename VectorOut, typename MatrixIn,
          typename MatrixOut>
Status FuzzyCTCLossCalculator::CalculateLoss(
    const VectorIn& seq_len, const LabelSequences& labels,
    const std::vector<MatrixIn>& inputs,
    bool ignore_longer_outputs_than_inputs,
    VectorOut* loss, std::vector<MatrixOut>* gradients,
    DeviceBase::CpuWorkerThreads* workers) const {

  if (loss == nullptr) {
    return errors::InvalidArgument("loss == nullptr");
  }

  bool requires_backprop = (gradients != nullptr);

  auto batch_size = inputs.size();
  auto num_classes = inputs[0].cols();

  if (loss->size() != batch_size) {
    return errors::InvalidArgument("loss.size() != batch_size");
  }
  loss->setZero();


  // Check validity of sequence_length array values.
  auto max_seq_len = seq_len(0);
  for (int b = 0; b < batch_size; b++) {
    if (inputs[b].cols() != num_classes) {
      return errors::InvalidArgument("Expected class count at b: ", b,
                                     " to be: ", num_classes,
                                     " but got: ", inputs[b].cols());
    }
    if (seq_len(b) < 0) {
      return errors::InvalidArgument("seq_len(", b, ") < 0");
    }
    if (seq_len(b) > inputs[b].rows()) {
      return errors::InvalidArgument("seq_len(", b, ") > ", inputs[b].rows());
    }
    max_seq_len = std::max(seq_len(b), max_seq_len);
  }

  TargetSequences target_sequences(batch_size);
  size_t max_u_prime = 0;
  Status p_t_ret = PopulateTargets(
            false,
            ignore_longer_outputs_than_inputs,
            batch_size, num_classes,
            seq_len, labels, &max_u_prime,
            &target_sequences);

  if (!p_t_ret.ok()) {
      return p_t_ret;
  }

  // and calculate the maximum necessary allocation size.
  /*LabelSequences l_primes(batch_size);
  size_t max_u_prime = 0;
  Status l_p_ret = PopulateLPrimes(
      preprocess_collapse_repeated, ignore_longer_outputs_than_inputs,
      batch_size, num_classes, seq_len, labels, &max_u_prime, &l_primes);
  if (!l_p_ret.ok()) {
    return l_p_ret;
  }*/
  auto one_norm_rowwise = [](Matrix &m) {
      m.noalias() = (m.array() / (m.rowwise().sum() * Matrix::Ones(1, m.cols())).array()).matrix();
  };
  auto softmax_norm_rowwise = [one_norm_rowwise](Matrix &m) {
      auto full_max = m.rowwise().maxCoeff() * Matrix::Ones(1, m.cols());
      m.noalias() = (m - full_max).array().exp().matrix();
      one_norm_rowwise(m);
  };


  // Process each item in a batch in parallel, using at most kMaxThreads.
  auto ComputeLossAndGradients = [this, num_classes, &labels, &target_sequences,
                                  &seq_len, &inputs, requires_backprop,
                                  ignore_longer_outputs_than_inputs, &loss,
                                  &gradients, softmax_norm_rowwise](int64 start_row,
                                              int64 limit_row) {
    for (int b = start_row; b < limit_row; b++) {
      // Return zero gradient for empty sequences or sequences with labels
      // longer than input, which is not supported by CTC.
      if (seq_len(b) == 0 ||
          (ignore_longer_outputs_than_inputs &&
           labels[b].size() > seq_len(b))) {
        VLOG(1) << "The sequence length is either zero or shorter than the "
                   "target output (CTC works only with shorter target sequence "
                   "than input sequence). You can turn this into a warning by "
                   "using the flag ignore_longer_outputs_than_inputs - "
                << b << ": " << str_util::Join(labels[b], " ");
        continue;
      }

      // For each batch element, log(alpha) and log(beta).
      //   row size is: u_prime == l_prime.size()
      //   col size is: seq_len[b] - output_delay_
      const auto& targets = target_sequences[b];
      Matrix softmax = inputs[b];
      softmax_norm_rowwise(softmax);
      Matrix aligned;
      AlignTargets(softmax, targets, &aligned);
      Matrix grads;
      CalculateGradient(softmax, aligned, &grads);
      (*loss)(b) = grads.squaredNorm();
      if (requires_backprop) {
          gradients->at(b) = -grads;
      }
    }  // for (int b = ...
  };
  /*if (workers) {
    // *Rough* estimate of the cost for one item in the batch.
    // Forward, Backward: O(T * U (= 2L + 1)), Gradients: O(T * (U + L)).
    //
    // softmax: T * L * (Cost(Exp) + Cost(Div))softmax +
    // fwd,bwd: T * 2 * (2*L + 1) * (Cost(LogSumExp) + Cost(Log)) +
    // grad: T * ((2L + 1) * Cost(LogSumExp) + L * (Cost(Expf) + Cost(Add)).
    const int64 cost_exp = Eigen::internal::functor_traits<
        Eigen::internal::scalar_exp_op<float>>::Cost;
    const int64 cost_log = Eigen::internal::functor_traits<
        Eigen::internal::scalar_log_op<float>>::Cost;
    const int64 cost_log_sum_exp =
        Eigen::TensorOpCost::AddCost<float>() + cost_exp + cost_log;
    const int64 cost =
        max_seq_len * num_classes *
            (cost_exp + Eigen::TensorOpCost::DivCost<float>()) +
        max_seq_len * 2 * (2 * num_classes + 1) *
            (cost_log_sum_exp + cost_log) +
        max_seq_len *
            ((2 * num_classes + 1) * cost_log_sum_exp +
             num_classes * (cost_exp + Eigen::TensorOpCost::AddCost<float>()));
    Shard(workers->num_threads, workers->workers, batch_size, cost,
          ComputeLossAndGradients);
  } else {
  */
    ComputeLossAndGradients(0, batch_size);
  /*}*/
  return Status::OK();
}

template <typename Vector>
Status FuzzyCTCLossCalculator::PopulateTargets(
    bool preprocess_collapse_repeated, bool ignore_longer_outputs_than_inputs,
    int batch_size, int num_classes, const Vector& seq_len,
    const LabelSequences& labels, size_t* max_u_prime,
    TargetSequences* targets) const {
  // labels is a Label array of size batch_size
  if (labels.size() != batch_size) {
    return errors::InvalidArgument(
        "labels.size() != batch_size: ", labels.size(), " vs. ", batch_size);
  }

  *max_u_prime = 0;  // keep track of longest l' modified label sequence.
  for (int b = 0; b < batch_size; b++) {
    // Assume label is in Label proto
    const std::vector<int>& label = labels[b];
    if (label.size() == 0) {
      return errors::InvalidArgument("Labels length is zero in batch ", b);
    }

    // If debugging: output the labels coming into training.
    //
    VLOG(2) << "label for batch: " << b << ": " << str_util::Join(label, " ");

    // Target indices, length = U.
    std::vector<int> l;

    // Convert label from DistBelief
    bool finished_sequence = false;
    for (int i = 0; i < label.size(); ++i) {
      if (i == 0 || !preprocess_collapse_repeated || label[i] != label[i - 1]) {
        if (label[i] >= num_classes - 1) {
          finished_sequence = true;
        } else {
          if (finished_sequence) {
            // Saw an invalid sequence with non-null following null
            // labels.
            return errors::InvalidArgument(
                "Saw a non-null label (index >= num_classes - 1) "
                "following a ",
                "null label, batch: ", b, " num_classes: ", num_classes,
                " labels: ", str_util::Join(l, ","));
          }
          l.push_back(label[i]);
        }
      }
    }

    for (int l_i : l) {
      if (l_i < 0) {
        return errors::InvalidArgument(
            "All labels must be nonnegative integers, batch: ", b,
            " labels: ", str_util::Join(l, ","));
      } else if (l_i >= num_classes) {
        return errors::InvalidArgument(
            "No label may be greater than num_classes. ",
            "num_classes: ", num_classes, ", batch: ", b,
            " labels: ", str_util::Join(l, ","));
      }
    }
    if (!ignore_longer_outputs_than_inputs) {
      // Make sure there is enough time to output the target indices.
      int time = seq_len(b);
      int required_time = label.size();
      if (required_time > time) {
        return errors::InvalidArgument(
            "Not enough time for target transition sequence ("
            "required: ",
            required_time, ", available: ", time, ")", b,
            "You can turn this error into a warning by using the flag "
            "ignore_longer_outputs_than_inputs");
      }
    }
    // Target indices with blanks before each index and a blank at the end.
    // Length U' = 2U + 1.
    // Convert l to l_prime
    MakeTarget(l, num_classes, &targets->at(b));
    *max_u_prime = std::max<size_t>(*max_u_prime, targets->at(b).rows());
  }
  return Status::OK();
}

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_
