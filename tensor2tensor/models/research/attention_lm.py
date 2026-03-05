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

"""基于自注意力的语言模型。

已弃用。请使用支持仅解码器模式的 Transformer。

类似于 transformer.py，但没有编码器

decoder: [Self-Attention, Feed-forward] x n

该模型使用自注意力机制进行语言建模，
适用于文本生成和语言理解任务。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf

framework = contrib.framework(msg="warn")


@framework.deprecated(
    "2018-09-15", "Use Transformer, which supports decoder-only mode when "
    "Transformer.has_input=False.")
@registry.register_model
class AttentionLM(t2t_model.T2TModel):
  """注意力网络。

  基于自注意力的语言模型，现已弃用，建议使用 Transformer 的仅解码器模式。
  """

  def body(self, features):
    """模型主体函数。

    参数：
        features: 输入特征字典

    返回：
        模型输出
    """
    # Remove dropout if not training
    hparams = self._hparams
    targets = features["targets"]
    targets = tf.squeeze(targets, 2)

    (decoder_input, decoder_self_attention_bias) = attention_lm_prepare_decoder(
        targets, hparams)

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    decoder_output = attention_lm_decoder(decoder_input,
                                          decoder_self_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def attention_lm_prepare_decoder(targets, hparams):
  """准备解码器的一个分片。

  参数：
      targets: 目标张量
      hparams: 运行超参数

  返回：
      decoder_input: 张量，解码器栈的底部
      decoder_self_attention_bias: 张量，包含大的负值以实现掩码注意力，
          以及可能用于对角对齐的偏置
  """
  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepend_inputs_full_attention(
            common_attention.embedding_to_padding(targets)))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(
            common_layers.shape_list(targets)[1]))
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def attention_lm_decoder(decoder_input,
                         decoder_self_attention_bias,
                         hparams,
                         name="decoder"):
  """注意力语言模型的解码器栈。

  参数：
      decoder_input: 输入张量
      decoder_self_attention_bias: 自注意力的偏置张量
          （见 common_attention.attention_bias()）
      hparams: 模型超参数
      name: 字符串名称

  返回：
      y: 解码器输出张量
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams), None, decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = common_layers.conv_hidden_relu(
              common_layers.layer_preprocess(x, hparams),
              hparams.filter_size,
              hparams.hidden_size,
              dropout=hparams.relu_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
    return common_layers.layer_preprocess(x, hparams)


@registry.register_hparams
def attention_lm_base():
  """基础超参数集。

  返回：
      HParams 对象，包含注意力语言模型的基础超参数配置
  """
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 1024
  hparams.batch_size = 8192
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # 即无梯度裁剪
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 2000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = False

  hparams.add_hparam("filter_size", 4096)  # 像这样添加新的超参数
  # 注意力相关的标志
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # 所有以 "dropout" 结尾的超参数在非训练模式下会自动设置为 0.0
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("encoder_full_attention", False)
  return hparams


@registry.register_hparams
def attention_lm_small():
  """轻量级模型。

  在 lm1b_32k 上：
     45M 参数
     在 [GeForce GTX TITAN X] 上 2 steps/sec

  返回：
      HParams 对象，包含轻量级注意力语言模型的超参数配置
  """
  hparams = attention_lm_base()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.layer_prepostprocess_dropout = 0.5
  return hparams


@registry.register_hparams
def attention_lm_translation():
  """用于序列到序列任务的版本。

  返回：
      HParams 对象，包含用于序列到序列任务的注意力语言模型超参数配置
  """
  hparams = attention_lm_base()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.learning_rate = 0.4
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 512
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams
def attention_lm_translation_l12():
  """用于序列到序列任务的 12 层版本。

  返回：
      HParams 对象，包含 12 层序列到序列注意力语言模型的超参数配置
  """
  hparams = attention_lm_translation()
  hparams.batch_size = 4096
  hparams.num_hidden_layers = 12
  return hparams


@registry.register_hparams
def attention_lm_translation_full_attention():
  """使用全注意力的序列到序列版本。

  返回：
      HParams 对象，包含使用全注意力的序列到序列注意力语言模型超参数配置
  """
  hparams = attention_lm_translation()
  hparams.prepend_mode = "prepend_inputs_full_attention"
  return hparams
