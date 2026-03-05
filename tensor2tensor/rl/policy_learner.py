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

"""不同 RL 算法的统一接口。

为各种强化学习算法提供统一的训练和评估接口，
支持模型自由和基于模型的强化学习方法。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PolicyLearner(object):
  """策略学习器 API。

  定义强化学习策略学习器的标准接口，包括训练和评估方法。
  """

  def __init__(
      self, frame_stack_size, base_event_dir, agent_model_dir, total_num_epochs
  ):
    """初始化策略学习器。

    参数：
        frame_stack_size: 帧堆叠大小
        base_event_dir: 基础事件目录
        agent_model_dir: 代理模型目录
        total_num_epochs: 总训练轮次数
    """
    self.frame_stack_size = frame_stack_size
    self.base_event_dir = base_event_dir
    self.agent_model_dir = agent_model_dir
    self.total_num_epochs = total_num_epochs

  def train(
      self,
      env_fn,
      hparams,
      simulated,
      save_continuously,
      epoch,
      sampling_temp=1.0,
      num_env_steps=None,
      env_step_multiplier=1,
      eval_env_fn=None,
      report_fn=None
  ):
    """训练策略。

    参数：
        env_fn: 环境创建函数
        hparams: 超参数对象
        simulated: 是否使用模拟环境
        save_continuously: 是否持续保存模型
        epoch: 当前轮次
        sampling_temp: 采样温度，默认为 1.0
        num_env_steps: 环境步数，可选
        env_step_multiplier: 环境步数乘数，默认为 1
        eval_env_fn: 评估环境创建函数，可选
        report_fn: 报告函数，可选
    """
    raise NotImplementedError()

  def evaluate(self, env_fn, hparams, sampling_temp):
    """评估策略。

    参数：
        env_fn: 环境创建函数
        hparams: 超参数对象
        sampling_temp: 采样温度
    """
