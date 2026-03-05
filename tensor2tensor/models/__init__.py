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

"""T2T 中定义的模型。此处的导入会强制注册模型。

功能说明：
- 导入并注册所有 T2T 模型
- 包含 Transformer、ResNet、LSTM 等经典架构
- 支持图像、文本、视频等多种模态的模型
- 提供研究性模型和实验性模型
- 通过 registry 机制实现模型的统一管理
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

# pylint: disable=unused-import

from tensor2tensor.layers import modalities  # pylint: disable=g-import-not-at-top
from tensor2tensor.models import basic
from tensor2tensor.models import bytenet
from tensor2tensor.models import distillation
from tensor2tensor.models import evolved_transformer
from tensor2tensor.models import image_transformer
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.models import lstm
from tensor2tensor.models import neural_assistant
from tensor2tensor.models import neural_gpu
from tensor2tensor.models import resnet
from tensor2tensor.models import revnet
from tensor2tensor.models import shake_shake
from tensor2tensor.models import slicenet
from tensor2tensor.models import text_cnn
from tensor2tensor.models import transformer
from tensor2tensor.models import vanilla_gan
from tensor2tensor.models import xception
from tensor2tensor.models.neural_architecture_search import nas_model
from tensor2tensor.models.research import adafactor_experiments
from tensor2tensor.models.research import aligned
from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import cycle_gan
from tensor2tensor.models.research import gene_expression
from tensor2tensor.models.research import neural_stack
from tensor2tensor.models.research import residual_shuffle_exchange
from tensor2tensor.models.research import rl
from tensor2tensor.models.research import shuffle_network
from tensor2tensor.models.research import similarity_transformer
from tensor2tensor.models.research import super_lm
from tensor2tensor.models.research import transformer_moe
from tensor2tensor.models.research import transformer_nat
from tensor2tensor.models.research import transformer_parallel
from tensor2tensor.models.research import transformer_revnet
from tensor2tensor.models.research import transformer_seq2edits
from tensor2tensor.models.research import transformer_sketch
from tensor2tensor.models.research import transformer_symshard
from tensor2tensor.models.research import transformer_vae
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models.video import basic_deterministic
from tensor2tensor.models.video import basic_recurrent
from tensor2tensor.models.video import basic_stochastic
from tensor2tensor.models.video import emily
from tensor2tensor.models.video import savp
from tensor2tensor.models.video import sv2p
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

# The following models can't be imported under TF2
if not contrib.is_tf2:
  # pylint: disable=g-import-not-at-top
  from tensor2tensor.models.research import attention_lm
  from tensor2tensor.models.research import attention_lm_moe
  from tensor2tensor.models.research import glow
  from tensor2tensor.models.research import lm_experiments
  from tensor2tensor.models.research import moe_experiments
  from tensor2tensor.models.research import multiquery_paper
  from tensor2tensor.models import mtf_image_transformer
  from tensor2tensor.models import mtf_resnet
  from tensor2tensor.models import mtf_transformer
  from tensor2tensor.models import mtf_transformer2
  from tensor2tensor.models.research import vqa_attention
  from tensor2tensor.models.research import vqa_recurrent_self_attention
  from tensor2tensor.models.research import vqa_self_attention
  from tensor2tensor.models.video import epva
  from tensor2tensor.models.video import next_frame_glow
  # pylint: enable=g-import-not-at-top

# ========== 导入并注册所有模型 ==========
# pylint: disable=unused-import

# 导入模态层（用于处理不同类型的输入/输出）
from tensor2tensor.layers import modalities  # pylint: disable=g-import-not-at-top

# ========== 核心模型 ==========
from tensor2tensor.models import basic  # 基础模型
from tensor2tensor.models import bytenet  # ByteNet 模型
from tensor2tensor.models import distillation  # 知识蒸馏模型
from tensor2tensor.models import evolved_transformer  # 进化 Transformer
from tensor2tensor.models import image_transformer  # 图像 Transformer
from tensor2tensor.models import image_transformer_2d  # 2D 图像 Transformer
from tensor2tensor.models import lstm  # LSTM 模型
from tensor2tensor.models import neural_assistant  # 神经助手模型
from tensor2tensor.models import neural_gpu  # Neural GPU 模型
from tensor2tensor.models import resnet  # ResNet 残差网络
from tensor2tensor.models import revnet  # RevNet 可逆网络
from tensor2tensor.models import shake_shake  # Shake-Shake 网络
from tensor2tensor.models import slicenet  # SliceNet 模型
from tensor2tensor.models import text_cnn  # 文本 CNN
from tensor2tensor.models import transformer  # Transformer 模型
from tensor2tensor.models import vanilla_gan  # Vanilla GAN
from tensor2tensor.models import xception  # Xception 网络

# ========== 神经架构搜索模型 ==========
from tensor2tensor.models.neural_architecture_search import nas_model  # NAS 模型

# ========== 研究性模型 ==========
from tensor2tensor.models.research import adafactor_experiments  # Adafactor 实验
from tensor2tensor.models.research import aligned  # 对齐模型
from tensor2tensor.models.research import autoencoders  # 自编码器
from tensor2tensor.models.research import cycle_gan  # CycleGAN
from tensor2tensor.models.research import gene_expression  # 基因表达模型
from tensor2tensor.models.research import neural_stack  # 神经栈
from tensor2tensor.models.research import residual_shuffle_exchange  # 残差洗牌交换
from tensor2tensor.models.research import rl  # 强化学习模型
from tensor2tensor.models.research import shuffle_network  # 洗牌网络
from tensor2tensor.models.research import similarity_transformer  # 相似度 Transformer
from tensor2tensor.models.research import super_lm  # 超级语言模型
from tensor2tensor.models.research import transformer_moe  # Transformer MoE（混合专家）
from tensor2tensor.models.research import transformer_nat  # Transformer NAT
from tensor2tensor.models.research import transformer_parallel  # 并行 Transformer
from tensor2tensor.models.research import transformer_revnet  # Transformer-RevNet
from tensor2tensor.models.research import transformer_seq2edits  # Transformer Seq2Edits
from tensor2tensor.models.research import transformer_sketch  # Transformer Sketch
from tensor2tensor.models.research import transformer_symshard  # 对称分片 Transformer
from tensor2tensor.models.research import transformer_vae  # Transformer VAE
from tensor2tensor.models.research import universal_transformer  # 通用 Transformer

# ========== 视频预测模型 ==========
from tensor2tensor.models.video import basic_deterministic  # 基本确定性视频预测
from tensor2tensor.models.video import basic_recurrent  # 基本循环视频预测
from tensor2tensor.models.video import basic_stochastic  # 基本随机视频预测
from tensor2tensor.models.video import emily  # Emily 视频模型
from tensor2tensor.models.video import savp  # SAVP 视频模型
from tensor2tensor.models.video import sv2p  # SV2P 视频预测

# 导入 contrib 工具（用于 TF1/TF2 兼容性检查）
from tensor2tensor.utils import contrib
# 导入注册表（用于模型注册和查询）
from tensor2tensor.utils import registry


def model(name):
  """按名称获取模型类。
  
  Args:
    name: 模型名称（字符串）
  
  Returns:
    T2TModel 的子类
  
  功能说明：
  - 通过 registry 查询注册的模型
  - 支持所有已导入的模型类型
  """
  return registry.model(name)
