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

"""ByteNet 模型。

ByteNet 是一种基于膨胀卷积（dilated convolution）的序列到序列模型，
使用残差连接和膨胀卷积来捕获长距离依赖关系。

功能说明：
- 实现 ByteNet 架构（基于膨胀卷积的 seq2seq 模型）
- 支持残差连接加速训练
- 使用膨胀卷积捕获长距离依赖
- 适用于机器翻译和序列标注任务
- 提供比 RNN 更快的并行计算能力
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf


def residual_dilated_conv(x, repeat, padding, name, hparams):
  """带有残差连接的膨胀卷积块堆栈。
  
  Args:
    x: 输入张量
    repeat: 残差块重复次数
    padding: 卷积填充方式（'SAME'或'VALID'）
    name: TensorFlow 变量作用域名称
    hparams: 超参数对象，包含 kernel_height、kernel_width 等配置
  
  Returns:
    经过残差卷积处理后的输出张量
  
  功能说明：
  - 实现多层膨胀卷积的残差块
  - 使用指数增长的膨胀率（2^i）捕获多尺度特征
  - 应用层归一化稳定训练
  - 残差连接缓解梯度消失问题
  """
  with tf.variable_scope(name):
    # 设置卷积核尺寸
    k = (hparams.kernel_height, hparams.kernel_width)
    # 创建膨胀率和卷积核的配置列表
    # 膨胀率按 2^i 指数增长：1, 2, 4, 8, ...
    dilations_and_kernels = [((2**i, 1), k)
                             for i in range(hparams.num_hidden_layers)]
    
    # 重复指定次数的残差块
    for i in range(repeat):
      with tf.variable_scope("repeat_%d" % i):
        # 先应用层归一化
        y = common_layers.conv_block(
            common_layers.layer_norm(x, hparams.hidden_size, name="lnorm"),
            hparams.hidden_size,
            dilations_and_kernels,
            padding=padding,
            name="residual_conv")
        # 应用 dropout 正则化
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        # 残差连接：将输出加到输入上
        x += y
    return x


def bytenet_internal(inputs, targets, hparams):
  """ByteNet 的主要步骤，用于训练。
  
  Args:
    inputs: 输入序列张量
    targets: 目标序列张量
    hparams: 超参数对象
  
  Returns:
    模型输出（logits）
  
  功能说明：
  - 实现 ByteNet 的训练流程
  - 处理输入和目标序列
  - 应用膨胀卷积和残差连接
  - 生成预测输出
  """
  with tf.variable_scope("bytenet"):
    # 展平输入并将长度扩展 50%
    # Flatten inputs and extend length by 50%.
    inputs = tf.expand_dims(common_layers.flatten4d3d(inputs), axis=2)
    extend_length = tf.to_int32(0.5 * tf.to_float(tf.shape(inputs)[1]))
    inputs_shape = inputs.shape.as_list()
    inputs = tf.pad(inputs, [[0, 0], [0, extend_length], [0, 0], [0, 0]])
    inputs_shape[1] = None
    inputs.set_shape(inputs_shape)  # Don't lose the other shapes when padding.
    # Pad inputs and targets to be the same length, divisible by 50.
    inputs, targets = common_layers.pad_to_same_length(
        inputs, targets, final_length_divisible_by=50)
    final_encoder = residual_dilated_conv(inputs, hparams.num_block_repeat,
                                          "SAME", "encoder", hparams)

    shifted_targets = common_layers.shift_right(targets)
    kernel = (hparams.kernel_height, hparams.kernel_width)
    decoder_start = common_layers.conv_block(
        tf.concat([final_encoder, shifted_targets], axis=3),
        hparams.hidden_size, [((1, 1), kernel)],
        padding="LEFT")

    return residual_dilated_conv(decoder_start, hparams.num_block_repeat,
                                 "LEFT", "decoder", hparams)


@registry.register_model
class ByteNet(t2t_model.T2TModel):
  """ByteNet 模型类。

  实现 ByteNet 序列到序列模型，使用膨胀卷积和残差连接。
  """

  def body(self, features):
    """模型主体函数。

    参数：
        features: 输入特征字典

    返回：
        模型输出
    """
    return bytenet_internal(features["inputs"], features["targets"],
                            self._hparams)


@registry.register_hparams
def bytenet_base():
  """超参数集。

  定义 ByteNet 模型的基础超参数配置。

  返回：
      HParams 对象，包含 ByteNet 模型的超参数配置
  """
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 2048
  hparams.hidden_size = 768
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 4
  hparams.kernel_height = 3
  hparams.kernel_width = 1
  hparams.learning_rate_decay_scheme = "exp"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("num_block_repeat", 4)
  return hparams
