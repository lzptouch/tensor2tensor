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

"""T2T 超参数处理。

提供超参数的创建、加载、保存和操作工具函数。

功能说明：
- 提供超参数对象的创建和复制功能
- 支持从 JSON 文件加载超参数
- 支持超参数覆盖和修改
- 集成问题相关的超参数配置
- 提供超参数的序列化和反序列化
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def copy_hparams(hparams):
  """复制超参数对象。
  
  Args:
    hparams: 要复制的超参数对象
  
  Returns:
    新的超参数对象（深拷贝）
  
  功能说明：
  - 创建超参数对象的独立副本
  - 复制所有超参数值
  - 保留 problem 和 problem_hparams 属性
  - 用于实验对比和配置备份
  """
  # 获取超参数的所有值
  hp_vals = hparams.values()
  # 使用这些值创建新的 HParams 对象
  new_hparams = hparam.HParams(**hp_vals)
  # 复制其他特殊属性（problem 和 problem_hparams）
  other_attrs = ["problem", "problem_hparams"]
  for attr in other_attrs:
    attr_val = getattr(hparams, attr, None)
    if attr_val is not None:
      setattr(new_hparams, attr, attr_val)
  return new_hparams


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None,
                   hparams_path=None):
  """创建包含数据目录和问题超参数的 HParams。
  
  Args:
    hparams_set: 超参数集名称（从 registry 获取）
    hparams_overrides_str: 超参数覆盖字符串（逗号分隔的键值对）
    data_dir: 数据目录路径
    problem_name: 问题名称（用于加载问题特定配置）
    hparams_path: 可选的 JSON 格式超参数文件路径
  
  Returns:
    配置完整的超参数对象
  
  功能说明：
  - 从 registry 获取基础超参数配置
  - 支持从 JSON 文件加载超参数
  - 支持命令行覆盖超参数值
  - 自动添加数据目录和问题相关配置
  """
  # 从注册表获取基础超参数配置
  hparams = registry.hparams(hparams_set)
  
  # 如果提供了 JSON 文件路径，从中加载超参数
  if hparams_path and tf.gfile.Exists(hparams_path):
    hparams = create_hparams_from_json(hparams_path, hparams)
  
  # 添加数据目录配置
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  
  # 应用超参数覆盖（来自命令行）
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  
  # 添加问题相关的超参数
  if problem_name:
    add_problem_hparams(hparams, problem_name)
  
  return hparams


def create_hparams_from_json(json_path, hparams=None):
  """从 JSON 文件加载超参数配置。
  
  Args:
    json_path: JSON 文件路径
    hparams: 可选的初始超参数对象（用于合并）
  
  Returns:
    加载后的超参数对象
  
  功能说明：
  - 从 JSON 文件反序列化超参数
  - 支持与现有超参数对象合并
  - 过滤掉函数类型的特殊键（bottom、loss 等）
  - 记录覆盖的超参数值
  """
  tf.logging.info("Loading hparams from existing json %s" % json_path)
  with tf.gfile.Open(json_path, "r") as f:
    # 加载 JSON 格式的超参数值
    hparams_values = json.load(f)
    # 防止某些特殊键覆盖传入的 hparams
    # TODO(trandustin): 在 registry 可用后移除此 hack，避免保存为函数
    if hparams:
      # 移除函数类型的键（这些不应该被保存和覆盖）
      hparams_values.pop("bottom", None)
      hparams_values.pop("loss", None)
      hparams_values.pop("name", None)
      hparams_values.pop("top", None)
      hparams_values.pop("weights_fn", None)
    # 创建新的 HParams 对象
    new_hparams = hparam.HParams(**hparams_values)
    
    # 如果提供了初始 hparams，需要合并两个对象
    # 某些键在新对象中但不在旧对象中，所以不能简单地使用 parse_json()
    if hparams:  # 指定了 hparams，从 JSON 更新值
      for key in sorted(new_hparams.values().keys()):
        if hasattr(hparams, key):  # 重叠的键
          value = getattr(hparams, key)
          new_value = getattr(new_hparams, key)
          if value != new_value:  # 值不同，记录覆盖
            tf.logging.info("Overwrite key %s: %s -> %s" % (
                key, value, new_value))
            setattr(hparams, key, new_value)
    else:
      # 没有提供初始 hparams，直接使用新加载的对象
      hparams = new_hparams

  return hparams


def add_problem_hparams(hparams, problem_name_or_instance):
  """为超参数对象添加问题相关的配置。
  
  Args:
    hparams: 超参数对象
    problem_name_or_instance: 问题名称（字符串）或 Problem 实例
  
  功能说明：
  - 根据问题名称或实例获取问题特定超参数
  - 将 Problem 对象和 problem_hparams 添加到 hparams
  - 支持从 registry 查询或直接使用 Problem 实例
  """
  # 判断是 Problem 实例还是问题名称
  if isinstance(problem_name_or_instance, problem_lib.Problem):
    # 直接使用 Problem 实例
    problem = problem_name_or_instance
  else:
    # 从 registry 查询 Problem
    problem = registry.problem(problem_name_or_instance)
  
  # 获取问题的超参数配置
  p_hparams = problem.get_hparams(hparams)
  # 将 Problem 对象添加到 hparams
  hparams.problem = problem
  # 将问题特定的超参数添加到 hparams
  hparams.problem_hparams = p_hparams
