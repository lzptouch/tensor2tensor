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

"""工具函数模块。

包含用于训练、评估、数据处理等的各种工具函数。

功能说明：
- 提供训练流程管理（trainer_lib）
- 提供模型基类（t2t_model）
- 提供优化器实现（optimize、adafactor）
- 提供解码和束搜索（decoding、beam_search）
- 提供评估指标（metrics、bleu_hook、rouge）
- 提供超参数管理（hparam、hparams_lib）
- 提供设备管理（devices）
- 提供注册表机制（registry）
"""

