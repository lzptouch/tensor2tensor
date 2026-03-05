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

"""Adafactor 优化器的实验。

实现 Adafactor 优化器的各种实验配置，
探索其在 Transformer 模型上的表现。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


def mimic_adam_with_adafactor(hparams):
  """从 Adam 切换到 Adafactor，近似 Adam 的行为。

  一些细微的差异可能不同，比如 epsilon 和 beta1 校正。

  参数：
      hparams: 模型超参数，其中 hparams.optimizer 包含"adam"
  """
  assert "adam" in hparams.optimizer
  hparams.optimizer = "adafactor"
  hparams.optimizer_adafactor_beta1 = hparams.optimizer_adam_beta1
  hparams.optimizer_adafactor_beta2 = hparams.optimizer_adam_beta2
  hparams.optimizer_adafactor_multiply_by_parameter_scale = False
  hparams.optimizer_adafactor_factored = False
  hparams.optimizer_adafactor_clipping_threshold = None
  hparams.optimizer_adafactor_decay_type = "adam"


@registry.register_hparams
def afx_adam():
  """旧版本 - Adam 优化器。

  返回：
      HParams 对象，包含 Adam 优化器的超参数配置
  """
  hparams = transformer.transformer_base_v2()
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.symbol_modality_num_shards = 1
  hparams.batch_size = 2048
  hparams.optimizer = "adam"
  hparams.learning_rate_schedule = (
      "constant*rsqrt_decay*linear_warmup*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def afx_mimic_adam():
  """模拟 Adam - 应该与 afx_adam 非常相似。

  返回：
      HParams 对象，包含模拟 Adam 的 Adafactor 配置
  """
  hparams = afx_adam()
  mimic_adam_with_adafactor(hparams)
  return hparams


@registry.register_hparams
def afx_base():
  """基线 - 无动量，beta=0.999。

  返回：
      HParams 对象，包含基线 Adafactor 配置
  """
  hparams = afx_mimic_adam()
  hparams.optimizer_adafactor_beta1 = 0.0
  return hparams


@registry.register_hparams
def afx_factored():
  """使用分解形式的 Adafactor。

  返回：
      HParams 对象，包含分解形式的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_factored = True
  return hparams


@registry.register_hparams
def afx_fast():
  """快速版本的 Adafactor - beta2=0.9。

  返回：
      HParams 对象，包含快速版本的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_beta2 = 0.9
  return hparams


@registry.register_hparams
def afx_clip():
  """带裁剪的 Adafactor - 裁剪阈值为 1.0。

  返回：
      HParams 对象，包含带裁剪的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_clipping_threshold = 1.0
  return hparams


@registry.register_hparams
def afx_clip2():
  """带裁剪的 Adafactor - 裁剪阈值为 2.0。

  返回：
      HParams 对象，包含带裁剪的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_clipping_threshold = 2.0
  return hparams


@registry.register_hparams
def afx_clip_factored():
  """带裁剪的分解形式 Adafactor。

  返回：
      HParams 对象，包含带裁剪的分解形式 Adafactor 配置
  """
  hparams = afx_clip()
  hparams.optimizer_adafactor_factored = True
  return hparams


@registry.register_hparams
def afx_pow05():
  """使用幂次衰减的 Adafactor - 内存指数为 0.5。

  返回：
      HParams 对象，包含幂次衰减的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_decay_type = "pow"
  hparams.optimizer_adafactor_memory_exponent = 0.5
  return hparams


@registry.register_hparams
def afx_pow08():
  """使用幂次衰减的 Adafactor - 内存指数为 0.8。

  返回：
      HParams 对象，包含幂次衰减的 Adafactor 配置
  """
  hparams = afx_pow05()
  hparams.optimizer_adafactor_memory_exponent = 0.8
  return hparams


@registry.register_hparams
def afx_pow10():
  """使用幂次衰减的 Adafactor - 内存指数为 1.0。

  返回：
      HParams 对象，包含幂次衰减的 Adafactor 配置
  """
  hparams = afx_pow05()
  hparams.optimizer_adafactor_memory_exponent = 1.0
  return hparams


@registry.register_hparams
def afx_pow08_clip():
  """带裁剪的幂次衰减 Adafactor - 内存指数为 0.8，裁剪阈值为 1.0。

  返回：
      HParams 对象，包含带裁剪的幂次衰减 Adafactor 配置
  """
  hparams = afx_pow08()
  hparams.optimizer_adafactor_clipping_threshold = 1.0
  return hparams


@registry.register_hparams
def afx_relative():
  """使用相对参数尺度的 Adafactor。

  返回：
      HParams 对象，包含使用相对参数尺度的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.optimizer_adafactor_multiply_by_parameter_scale = True
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def afx_unscale():
  """不使用嵌入缩放的 Adafactor。

  返回：
      HParams 对象，包含不使用嵌入缩放的 Adafactor 配置
  """
  hparams = afx_base()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.multiply_embedding_mode = "none"
  return hparams


@registry.register_hparams
def afx_unscale_relative():
  """不使用嵌入缩放且使用相对参数尺度的 Adafactor。

  返回：
      HParams 对象，包含不使用嵌入缩放且使用相对参数尺度的 Adafactor 配置
  """
  hparams = afx_unscale()
  hparams.optimizer_adafactor_multiply_by_parameter_scale = True
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def afx_adafactor():
  """使用推荐学习率调度的 Adafactor。

  返回：
      HParams 对象，包含使用推荐学习率调度的 Adafactor 配置
  """
  hparams = afx_adam()
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def afx_small():
  """小批量大小的小 Transformer 模型，用于快速步进。

  返回：
      HParams 对象，包含小批量大小的小 Transformer 模型配置
  """
  hparams = transformer.transformer_tpu()
  hparams.filter_size = 1024
  hparams.num_heads = 4
  hparams.num_hidden_layers = 3
  hparams.batch_size = 512
  return hparams


@registry.register_hparams
def afx_small_p16():
  """16 位量化的小 Transformer 模型。

  返回：
      HParams 对象，包含 16 位量化的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.add_hparam("simulated_quantize_bits", 16)
  return hparams


@registry.register_hparams
def afx_small_p12():
  """12 位量化的小 Transformer 模型。

  返回：
      HParams 对象，包含 12 位量化的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.add_hparam("simulated_parameter_quantize_bits", 12)
  return hparams


@registry.register_hparams
def afx_small_p11():
  """11 位量化的小 Transformer 模型。

  返回：
      HParams 对象，包含 11 位量化的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.add_hparam("simulated_parameter_quantize_bits", 11)
  return hparams


@registry.register_hparams
def afx_small_p10():
  """10 位量化的小 Transformer 模型。

  返回：
      HParams 对象，包含 10 位量化的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.add_hparam("simulated_parameter_quantize_bits", 10)
  return hparams


@registry.register_hparams
def afx_small_p8():
  """8 位量化的小 Transformer 模型。

  返回：
      HParams 对象，包含 8 位量化的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.add_hparam("simulated_parameter_quantize_bits", 8)
  return hparams


@registry.register_hparams
def afx_small_bfloat16():
  """使用 bfloat16 精度的小 Transformer 模型。

  返回：
      HParams 对象，包含使用 bfloat16 精度的小 Transformer 模型配置
  """
  hparams = afx_small()
  hparams.weight_dtype = "bfloat16"
  hparams.activation_dtype = "bfloat16"
  return hparams
