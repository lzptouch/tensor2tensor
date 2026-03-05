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

"""用于在较大数据批次上运行 Glow 初始化的钩子。

实现 Glow 模型的数据依赖初始化钩子，
在训练开始前使用较大数据批次进行初始化。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class GlowInitHook(tf.train.SessionRunHook):
  """Glow 初始化钩子。

  在第一步之前运行一次数据依赖的初始化。

  初始化操作存储在 tf 集合 glow_init_op 中。
  有关更多详细信息，请参阅 glow.py 中的"body"。
  """

  def after_create_session(self, session, coord):
    """在会话创建后运行。

    参数：
        session: TensorFlow 会话
        coord: 协调器
    """
    del coord
    global_step = session.run(tf.train.get_global_step())
    if global_step == 0:
      ddi = tf.get_collection("glow_init_op")
      # In-case of a multi-GPU system, this just runs the first op in the
      # collection.
      if ddi:
        session.run(ddi[0])
