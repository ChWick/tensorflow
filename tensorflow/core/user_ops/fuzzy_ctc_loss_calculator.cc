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

#include "tensorflow/core/user_ops/fuzzy_ctc_loss_calculator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace ctc {

// Calculates the alpha(t, u) as described in (GravesTh) Section 7.3.
// Starting with t = 0 instead of t = 1 used in the text.
// Based on Breuel's CTC.
void FuzzyCTCLossCalculator::CalculateForwardVariables(
        const MatrixXd &log_match, MatrixXd *forward) const {
    *forward = MatrixXd::Zero(log_match.rows(), log_match.cols());
    RowVectorXd v(log_match.cols()), w(log_match.cols());
    for (int i = 0; i < v.size(); ++i) {
        v(i) = i * skip_penalty_;
    }

    for (int t = 0; t < log_match.rows(); ++t) {
        for (int i = 1; i < log_match.cols(); ++i) {
            w(i) = v(i - 1);
        }
        w(0) = t * skip_penalty_;

        auto v1 = v + log_match.row(t);
        auto v2 = w + log_match.row(t);
        for (int i = 0; i < v1.size(); ++i) {
            v(i) = LogSumExp(v1(i), v2(i));
        }
        forward->row(t) = v;
    }
}

void FuzzyCTCLossCalculator::CalculateGradient(const Matrix &inputs, const Matrix &aligned, Matrix* dy) const {
    *dy = (aligned - inputs);
}

void FuzzyCTCLossCalculator::CalculateForwardBackward(const MatrixXd& log_match, MatrixXd* alpha_beta) const {
    MatrixXd alpha, beta;
    CalculateForwardVariables(log_match, &alpha);
    CalculateForwardVariables(log_match.reverse(), &beta);
    *alpha_beta = alpha + beta.reverse();
}

void FuzzyCTCLossCalculator::AlignTargets(const Matrix& inputs, const MatrixXd& targets, Matrix *aligned) const {
    auto p = [] (const std::string &s, const MatrixXd &m) {
        return;
        std::cerr << "\n" << s << "\n" << m.row(0) << "\n" << m.row(1) << "\n" << m.row(2) << "\n...\n"
            << m.row(m.rows() - 3) << "\n"
            << m.row(m.rows() - 2) << "\n"
            << m.row(m.rows() - 1) << "\n";
    };

    p("Targets", targets);

    auto one_norm_rowwise = [](MatrixXd &m) {
        m.noalias() = (m.array() / (m.rowwise().sum() * MatrixXd::Ones(1, m.cols())).array()).matrix();
    };
    auto one_norm_colwise = [](MatrixXd &m) {
        m.noalias() = (m.array() / (MatrixXd::Ones(m.rows(), 1) * m.colwise().sum()).array()).matrix();
    };
    MatrixXd outputs = inputs.cwiseMax(minimum_input_).cast<double>();
    one_norm_rowwise(outputs);
    p("Outputs", outputs);

    MatrixXd match = outputs * targets.transpose();
    match = match.array().log();
    p("lMatch", match);

    MatrixXd alpha_beta;
    CalculateForwardBackward(match, &alpha_beta);
    p("Alpha Beta", alpha_beta);

    alpha_beta.noalias() = (alpha_beta.array() - alpha_beta.maxCoeff()).exp().matrix();
    p("Alpha Beta exp", alpha_beta);
    one_norm_colwise(alpha_beta);
    p("Alpha Beta norm", alpha_beta);

    MatrixXd aligned_d = (alpha_beta * targets).cwiseMax(minimum_input_);
    p("Aligned", aligned_d);
    one_norm_rowwise(aligned_d);
    p("Aligned - norm", aligned_d);
    *aligned = aligned_d.cast<float>();
}

void FuzzyCTCLossCalculator::MakeTarget(const std::vector<int>& l,
        int num_classes,
        MatrixXd *target) const {
    MatrixXd &m = *target;
    m = MatrixXd::Zero(2 * l.size() + 1, num_classes);
    for (size_t i = 0; i < l.size(); ++i) {
        int c = l[i];
        m(2 * i, blank_index_) = 1.0;      // blank
        m(2 * i + 1, c) = 1.0;             // actual label      
    }

    m(2 * l.size(), blank_index_) = 1.0;              // last label
}

}  // namespace ctc
}  // namespace tensorflow
