# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""带有辅助损失的 Transformer。

来自 https://arxiv.org/abs/1803.00144 的带有辅助损失的 Transformer 模型。

该模型通过添加辅助损失来改进 Transformer 的训练，
这些辅助损失可以帮助模型学习更好的表示。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def shift_and_pad(tensor, shift, axis=0):
  """沿轴移动并用零填充。

  示例：
      shift_and_pad([1, 2, 3, 4], 2)  --> [0, 0, 1, 2]
      shift_and_pad([1, 2, 3, 4], -2) --> [3, 4, 0, 0]

  参数：
      tensor: 要移动和填充的张量
      shift: 整数，移动的步数
      axis: 整数，沿哪个轴移动和填充

  返回：
      与输入张量形状相同的张量
  """
  shape = tensor.shape
  rank = len(shape)
  assert 0 <= abs(axis) < rank

  length = int(shape[axis])
  assert 0 <= abs(shift) < length

  paddings = [(0, 0)] * rank
  begin = [0] * rank
  size = [-1] * rank

  if shift > 0:
    paddings[axis] = (shift, 0)
    size[axis] = length - shift
  elif shift < 0:
    paddings[axis] = (0, -shift)
    begin[axis] = -shift

  ret = tf.pad(tf.slice(tensor, begin, size), paddings)

  return ret


@registry.register_model
class TransformerAux(transformer.Transformer):
  """注意力网络。

  带有辅助损失的 Transformer 模型，
  通过辅助损失改进训练效果。
  """

  def _extract_shift_values(self):
    """解析 shift 字符串。

    超参数应包含 shift_values 键，映射到逗号分隔的整数字符串。
    这些整数指定要预测/重建以计算辅助损失的时间步数。

    例如，"-4,2,6"表示重建 4 步之前的目标，并预测
    2 步和 6 步之后的目标。

    返回：
        要计算辅助损失的移位值列表（非零整数）
    """
    shift_values_str = self._hparams.get("shift_values", "")
    shift_values = [int(x) for x in shift_values_str.split(",")]

    tf.logging.info(
        "Computing auxiliary losses for the following shifts: %s",
        shift_values)

    return shift_values

  def auxiliary_loss(self, body_output, features, shift):
    """辅助预测损失。

    参数：
        body_output: 形状为 [batch_size, decoder_length, hidden_dim] 的张量
        features: 模型的特征映射，必须包含：
            "targets": 目标解码器输出，形状为 [batch_size, decoder_length, 1, hidden_dim]
        shift: 非零整数，移动/填充目标序列的量。
            如果 shift > 0，表示要重建的先前时间步数；
            如果 shift < 0，表示要预测的未来时间步数

    返回：
        交叉熵损失的分子和分母的元组

    异常：
        ValueError: 如果 features 不包含 targets_raw 张量
    """
    assert isinstance(shift, int) and shift != 0
    name = "reconst_%d" % shift if shift > 0 else "predict_%d" % abs(shift)

    if features and "targets_raw" in features:
      targets = features["targets_raw"]
      targets = common_layers.flatten4d3d(targets)
    else:
      raise ValueError(
          "Feature map must contain a targets_raw tensor.")

    with tf.variable_scope(name):
      logits = self.top(body_output, features)
      labels = shift_and_pad(targets, shift, axis=1)
      return common_layers.padded_cross_entropy(
          logits,
          labels,
          self._hparams.label_smoothing)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target decoder outputs.
              [batch_size, target_length, 1, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      A 2-tuple containing:
          Logit tensor. [batch_size, decoder_length, vocab_size]
          Map of keys to loss tensors. Should contain the following:
              "training": Training loss (shift == 0).
              "auxiliary": Auxiliary loss (shift != 0).
    """
    output = super(TransformerAux, self).body(features)
    output, losses = self._normalize_body_output(output)

    aux = 0.0
    for shift in self._extract_shift_values():
      loss_num, loss_den = self.auxiliary_loss(output, features, shift)
      aux += loss_num / loss_den
    losses["auxiliary"] = aux

    return output, losses


@registry.register_hparams
def transformer_aux_base():
  """Set of hyperparameters."""
  hparams = transformer.transformer_base()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.add_hparam("shift_values", "1,2,3,4")
  return hparams


@registry.register_hparams
def transformer_aux_tiny():
  """Set of hyperparameters."""
  hparams = transformer.transformer_tiny()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.add_hparam("shift_values", "1,2")
  return hparams
