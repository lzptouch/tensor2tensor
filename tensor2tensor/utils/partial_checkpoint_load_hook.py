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

"""部分加载检查点的钩子。

用于从检查点部分加载模型权重的会话运行钩子实现。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class PartialCheckpointLoad(tf.train.SessionRunHook):
  """从检查点部分加载训练变量。

  用于将检查点中保存的每个变量加载到图中。
  它将忽略图中存在于检查点中未保存的额外变量。
  （注意：加载的变量包括 ADAM/训练变量，如果它们存在于检查点中）
  如果图中变量的基础作用域名与检查点变量不同，可以执行映射。
  """

  def __init__(self, hook_context, chk_scopename, graph_scopename):
    """使用检查点目录和作用域名初始化钩子。

    参数：
        hook_context: 包含超参数的 HookContext 对象
        chk_scopename: 正在加载的检查点中变量的基础作用域名
        graph_scopename: 当前图中变量的基础作用域名
    """
    self.checkpoint_path = hook_context.hparams.partial_load_checkpoint
    self.chk_scopename = chk_scopename
    self.graph_scopename = graph_scopename

  def begin(self):
    # TODO(karishmamalkan): Add logging for when variables are loaded
    variable_references = {var.name: var for var in tf.all_variables()}
    variable_mappings = {}
    vars_in_chk = tf.train.list_variables(self.checkpoint_path)
    for (var, _) in vars_in_chk:
      variable_mappings[var] = variable_references[
          var.replace(self.chk_scopename, self.graph_scopename) + ":0"]
    tf.train.init_from_checkpoint(self.checkpoint_path, variable_mappings)
