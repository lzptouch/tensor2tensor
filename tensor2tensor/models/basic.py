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

"""用于测试简单任务的基础模型。

包含简单的全连接网络等基础模型，用于快速测试和验证。

功能说明：
- 提供基础的全连接神经网络实现
- 支持 ReLU 激活函数
- 适用于小型数据集和快速原型验证
- 作为复杂模型的对比基线
- 支持多层堆叠和 dropout 正则化
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf


@registry.register_model
class BasicFcRelu(t2t_model.T2TModel):
  """基础的全连接 + ReLU 模型。
  
  用于简单任务的基础模型，包含多层全连接层和 ReLU 激活函数。
  
  功能说明：
  - 继承自 T2TModel 基类
  - 实现简单的全连接网络结构
  - 支持可配置的多隐藏层
  - 使用 ReLU 激活函数引入非线性
  - 应用 dropout 防止过拟合
  """

  def body(self, features):
    """模型主体函数：实现前向传播逻辑。
    
    Args:
      features: 输入特征字典，包含"inputs"键
    
    Returns:
      模型输出张量（4D 格式，适配 T2T）
    
    功能说明：
    - 将输入展平为一维向量
    - 通过多层全连接层和 ReLU 激活
    - 每层后应用 dropout 正则化
    - 恢复为 4D 格式以兼容 T2T 框架
    """
    hparams = self.hparams  # 获取超参数配置
    x = features["inputs"]  # 提取输入特征
    shape = common_layers.shape_list(x)  # 获取输入形状
    
    # 将 4D 输入 [batch, height, width, channels] 展平为 2D [batch, features]
    x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    
    # 堆叠多个全连接层
    for i in range(hparams.num_hidden_layers):
      # 全连接层：线性变换
      x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
      # 应用 dropout 正则化
      x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
      # ReLU 激活函数引入非线性
      x = tf.nn.relu(x)
    
    # 恢复为 4D 格式以兼容 T2T 框架
    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)


@registry.register_hparams
def basic_fc_small():
  """小型全连接模型的超参数配置。
  
  Returns:
    HParams 对象，包含模型的所有超参数
  
  功能说明：
  - 注册名为"basic_fc_small"的超参数配置
  - 设置较小的模型规模用于快速测试
  - 配置学习率、批量大小、隐藏层等参数
  - 禁用 dropout 和权重衰减（适用于小模型）
  """
  # 获取基础超参数配置
  hparams = common_hparams.basic_params1()
  # 设置学习率为 0.1（相对较大，适用于小模型）
  hparams.learning_rate = 0.1
  # 批量大小为 128
  hparams.batch_size = 128
  # 隐藏层单元数为 256
  hparams.hidden_size = 256
  # 隐藏层数量为 2 层
  hparams.num_hidden_layers = 2
  # 使用均匀单位缩放初始化器
  hparams.initializer = "uniform_unit_scaling"
  # 初始化增益为 1.0
  hparams.initializer_gain = 1.0
  # 禁用权重衰减（L2 正则化）
  hparams.weight_decay = 0.0
  # 禁用 dropout
  hparams.dropout = 0.0
  return hparams
