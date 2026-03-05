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

"""语言模型实验。

训练 languagemodel_lm1b32k_packed 并测量 log-ppl/token（开发集）。
这些数字需要乘以 1.107893 才能得到 log-ppl/word，以便与已发表的结果进行比较。

基本训练方案：300k 步 * 8 核心 * batch_size=4096
   = 约 10 个 epoch

确保在 CPU 或 GPU 上使用大量步数（1000）进行评估，因为
TPU 评估代码不知道如何在开发数据结束时停止。还需要
为评估设置 activation_type=float32，因为目前存在
daisy_chain_getter 和 activation_type=bfloat16 之间的冲突。

结果：
  lmx_base:      log-ppl/tok=3.40   PPL/word=43.2   (10 小时*8 核心)
  lmx_h1k_f4k:
  lmx_h2k_f8k:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


@registry.register_hparams
def lmx_base():
  """在 languagemodel_lm1b32k_packed 上使用 Transformer。50M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = transformer.transformer_tpu()
  # sharing is counterproductive when underparameterized
  hparams.shared_embedding_and_softmax_weights = False
  # we judge by log-ppl, so label smoothing hurts.
  hparams.label_smoothing = 0.0
  # This makes the batch size on GPU the same as on TPU for a packed problem
  # with sequence length 256.
  # TODO(noam): fix the mess that is the data reading pipeline.
  hparams.max_length = 256
  # larger batch since we only have a decoder
  hparams.batch_size = 4096
  # save some memory so we can have a larger model
  hparams.activation_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_h1k_f4k():
  """在 languagemodel_lm1b32k_packed 上使用 Transformer。140M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def lmx_h2k_f8k():
  """训练 languagemodel_lm1b32k_packed 的超参数。430M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams


@registry.register_hparams
def lmx_h3k_f12k():
  """训练 languagemodel_lm1b32k_packed 的超参数。880M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.hidden_size = 3072
  hparams.filter_size = 12288
  hparams.batch_size = 2048
  hparams.weight_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_h4k_f16k():
  """训练 languagemodel_lm1b32k_packed 的超参数。1470M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.hidden_size = 4096
  hparams.filter_size = 16384
  hparams.batch_size = 1024
  hparams.weight_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_relative():
  """使用相对注意力的语言模型。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.activation_dtype = "float32"
  hparams.weight_dtype = "float32"
  return hparams


@registry.register_hparams
def lmx_relative_nopos():
  """使用相对注意力且无位置编码的语言模型。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_relative()
  hparams.pos = "none"
  return hparams


@registry.register_hparams
def lmx_moe():
  """带有混合专家的 Transformer。140M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.ffn_layer = "local_moe_tpu"
  return hparams


@registry.register_hparams
def lmx_moe_h1k_f4k_x32():
  """带有混合专家的 Transformer。890M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_h1k_f4k()
  hparams.ffn_layer = "local_moe_tpu"
  hparams.moe_num_experts = 32
  hparams.weight_dtype = "bfloat16"
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def lmx_moe_h1k_f8k_x16():
  """带有混合专家的 Transformer。890M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_h1k_f4k()
  hparams.filter_size = 8192
  hparams.ffn_layer = "local_moe_tpu"
  hparams.moe_num_experts = 16
  hparams.weight_dtype = "bfloat16"
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def lmx_h1k_f64k():
  """训练 languagemodel_lm1b32k_packed 的超参数。880M 参数。

  返回：
      HParams 对象，包含语言模型训练的超参数配置
  """
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 65536
  hparams.batch_size = 2048
  return hparams
