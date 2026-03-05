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

"""Tensor2Tensor 命令行工具模块。

包含用于训练、数据生成、解码、评估等任务的命令行工具。

功能说明：
- 提供 t2t_trainer（主训练脚本）
- 提供 t2t_datagen（数据生成工具）
- 提供 t2t_decoder（解码和推理工具）
- 提供 t2t_eval（模型评估工具）
- 提供其他辅助工具（BLEU 评估、检查点平均等）
"""

