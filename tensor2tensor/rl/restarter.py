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

"""训练重启器。

用于处理训练过程中的重启，特别是在模型之间共享参数和检查点时非常有用。
"""

import contextlib
import os

import tensorflow.compat.v1 as tf


class Restarter(object):
  """处理训练重启。

  在模型之间共享参数（和检查点）时特别有用。

  参数：
      model_mode (str): 模型"模式"。不同模式有不同的本地步数计数器，
          但共享相同的全局步数计数器。也用于日志消息中。
      checkpoint_dir (str): 模型检查点目录。从最后一个检查点的名称推断全局步数。
      target_local_step (int): 训练模型的目标本地步数。

  属性：
      model_mode (str): 参见参数
      checkpoint_dir (str): 参见参数
      target_local_step (int): 参见参数
      target_global_step (int): 计算出的训练模型的目标全局步数
      should_skip (bool): 是否应该跳过训练，因为已完成的本地步数已超过目标。
          这在重启期间会发生。
      steps_to_go: 还需要走多少步。
      restarting (bool): 当前训练轮次是否被中断并正在重启。
  """

  def __init__(self, model_mode, checkpoint_dir, target_local_step):
    """初始化重启器。

    参数：
        model_mode: 模型模式
        checkpoint_dir: 检查点目录
        target_local_step: 目标本地步数
    """
    self.model_mode = model_mode
    self.checkpoint_dir = checkpoint_dir
    self.target_local_step = target_local_step
    self.target_global_step = None
    self.should_skip = False
    self.restarting = False

    self._counter_path = os.path.join(
        checkpoint_dir, "{}_step_counter".format(model_mode)
    )

    self._global_step = self._get_global_step()
    tf.logging.info(
        "Will load %s checkpoint %d", self.model_mode, self._global_step
    )

    (self._local_step_at_start, global_step_at_start) = self._read_counters()

    self.steps_to_go = target_local_step - self._local_step_at_start
    if self.steps_to_go <= 0:
      tf.logging.info(
          "Skipping training %s, requested %d steps, already done %d",
          self.model_mode, target_local_step, self._local_step_at_start
      )
      self.should_skip = True
      return

    if global_step_at_start != -1:
      # Restart.
      steps_done_this_epoch = self._global_step - global_step_at_start
      self.steps_to_go -= steps_done_this_epoch
      tf.logging.info(
          "Restarting training %s, %d steps already done this epoch",
          self.model_mode, steps_done_this_epoch
      )
      self.restarting = True

    self.target_global_step = self._global_step + self.steps_to_go

  @contextlib.contextmanager
  def training_loop(self):
    """Context manager wrapping the training loop, updates step counters."""
    if not self.restarting:
      self._write_counters(self._local_step_at_start, self._global_step)

    tf.logging.info(
        "Training %s up to %d, %d to go", self.model_mode,
        self.target_local_step, self.steps_to_go
    )

    yield

    self._write_counters(self.target_local_step, -1)

  def _get_global_step(self):
    checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
    if checkpoint:
      return int(checkpoint.split("-")[-1])
    else:
      return 0

  def _read_counters(self):
    try:
      with tf.gfile.Open(self._counter_path, "r") as f:
        return tuple(
            int(counter) for counter in f.read().split(" ")
        )
    except tf.errors.NotFoundError:
      return (0, -1)

  def _write_counters(self, local_step, global_step):
    with tf.gfile.Open(self._counter_path, "w") as f:
      f.write("{} {}".format(local_step, global_step))
