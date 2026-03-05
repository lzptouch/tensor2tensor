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

"""神经网络层模块。

包含用于构建深度学习模型的各种层，包括注意力层、卷积层、归一化层等。

功能说明：
- 提供通用层（common_layers）
- 提供注意力机制层（common_attention）
- 提供 Transformer 专用层（transformer_layers）
- 提供模态处理层（modalities）
- 提供超参数工具（common_hparams）
- 支持自定义层的扩展
"""

