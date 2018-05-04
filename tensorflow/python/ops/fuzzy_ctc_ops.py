from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_user_ops as _gen_user_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util.tf_export import tf_export

@tf_export('user_ops.fuzzy_ctc_loss')
def fuzzy_ctc_loss(labels, inputs, sequence_length,
                   ignore_longer_outputs_than_inputs=False,
                   time_major=True):
    if not isinstance(labels, sparse_tensor.SparseTensor):
        raise TypeError("Expected labels (first argument) to be a SparseTensor")

    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)

    loss, _ = _gen_user_ops.fuzzy_ctc_loss(inputs, labels.indices, labels.values, sequence_length)
    return loss

@ops.RegisterGradient("FuzzyCTCLoss")
def _FuzzyCTCLossGrad(op, grad_loss, _):
    grad_without_gradient = array_ops.prevent_gradient(
            op.outputs[1], message="Currently there is no way to take the second "
            " derivative of ctc_loss due to the fused implementation's interaction "
            " with tf.gradients()")
    return [_BroadcastMul(grad_loss, grad_without_gradient), None, None, None]

@tf_export('user_ops.fuzzy_ctc_greedy_decoder')
def fuzzy_ctc_greedy_decoder(inputs, sequence_length):
    outputs = fuzzy_module.fuzzy_ctc_greedy_decoder(inputs, sequence_length)
    (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
    return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)],
            log_probabilities)
