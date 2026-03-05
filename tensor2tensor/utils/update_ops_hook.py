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

"""运行 tf.GraphKeys.UPDATE_OPS 的钩子。

用于在训练过程中执行更新操作（如批归一化的移动平均更新）。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class UpdateOpsHook(tf.train.SessionRunHook):
  """运行 assign_ops 的钩子。

  用于在训练过程中执行更新操作，如批归一化层的移动平均更新。
  """

  def before_run(self, run_context):
    """在每次运行前获取更新操作。

    参数：
        run_context: 运行上下文

    返回：
        包含更新操作的 SessionRunArgs
    """
    del run_context
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    return tf.train.SessionRunArgs(update_ops)
