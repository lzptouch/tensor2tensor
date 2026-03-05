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

"""为教师 - 学生模型执行知识蒸馏。

此脚本旨在与 --model=distillation 一起使用。有关示例超参数和用法，请参阅模型。

如果只指定了 output_dir，则 teacher_dir 为 `output_dir/teacher`，
student_dir 为 `output_dir/student`。日志写入 output_dir 内。
如果还明确指定了 teacher_dir，则 student_dir 仍为 `output_dir/student`，
日志写入 output_dir。如果进一步指定了 student_dir，
则日志写入 student_dir，除非明确指定了 output_dir，此时 output_dir 仅包含日志。

功能说明：
- 实现知识蒸馏（Knowledge Distillation）技术
- 先训练教师模型（teacher model）
- 再用教师模型的输出指导学生模型（student model）训练
- 学生模型通常更小更快，适合部署
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "skip_teacher_training", False,
    "By default, we train teacher model. If set to True, skip the training.")
flags.DEFINE_string(
    "teacher_dir", None,
    "Directory to teacher network. If not specified, `output_dir/teacher` is "
    "used instead.")
flags.DEFINE_string(
    "student_dir", None,
    "Directory to student network. If not specified, `output_dir/student` is "
    "used instead.")


def main(argv):
  """主函数：执行知识蒸馏流程。
  
  Args:
    argv: 命令行参数列表
  
  流程说明：
  1. 初始化环境和导入模块
  2. 训练教师模型（如果未跳过）
  3. 使用教师模型指导学生模型训练
  4. 保存学生模型
  """
  # 设置日志级别
  tf.logging.set_verbosity(tf.logging.INFO)
  # 设置随机种子
  trainer_lib.set_random_seed(FLAGS.random_seed)
  # 导入用户自定义目录
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  # 检查是否需要显示注册表帮助并退出
  t2t_trainer.maybe_log_registry_and_exit()

  # 如果在 Google Cloud ML Engine 上运行，则启动云训练
  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  # 如果需要生成数据，则先执行数据生成
  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  # 获取云训练的输出目录
  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  # 处理未解析的命令行参数作为超参数
  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  # 根输出目录
  root_output_dir = FLAGS.output_dir

  # 确定教师模型的输出目录
  if FLAGS.teacher_dir:
    teacher_dir = FLAGS.teacher_dir
  else:
    # 默认在 output_dir/teacher 下
    teacher_dir = os.path.join(root_output_dir, "teacher")

  # Train Teacher ============
  # 训练教师模型阶段
  if FLAGS.skip_teacher_training:
    # 如果设置了跳过教师模型训练的标志
    tf.logging.info("training teacher skipped")
  else:
    # 创建教师模型的超参数
    hparams = t2t_trainer.create_hparams()
    # 设置蒸馏阶段为"train"（训练教师模型）
    hparams.distill_phase = "train"
    # 设置教师模型的输出目录
    FLAGS.output_dir = teacher_dir

    # 创建实验函数和运行配置
    exp_fn = t2t_trainer.create_experiment_fn()
    run_config = t2t_trainer.create_run_config(hparams)
    exp = exp_fn(run_config, hparams)
    
    # 如果是主节点（chief），保存元数据
    if t2t_trainer.is_chief():
      t2t_trainer.save_metadata(hparams)
    # 执行训练计划
    t2t_trainer.execute_schedule(exp)

  # ==========================
  # Train Student ============
  # 训练学生模型阶段
  hparams = t2t_trainer.create_hparams()
  # 添加教师模型目录作为超参数（学生模型需要从教师模型学习）
  hparams.add_hparam("teacher_dir", teacher_dir)
  # 设置蒸馏阶段为"distill"（蒸馏学生模型）
  hparams.distill_phase = "distill"
  
  # 确定学生模型的输出目录
  if FLAGS.student_dir:
    student_dir = FLAGS.student_dir
  else:
    # 默认在 output_dir/student 下
    student_dir = os.path.join(root_output_dir, "student")
  FLAGS.output_dir = student_dir
  # 添加学生模型目录作为超参数
  hparams.add_hparam("student_dir", student_dir)

  # 创建学生模型的实验函数和运行配置
  exp_fn = t2t_trainer.create_experiment_fn()
  run_config = t2t_trainer.create_run_config(hparams)
  exp = exp_fn(run_config, hparams)

  # 如果是主节点，保存元数据
  if t2t_trainer.is_chief():
    t2t_trainer.save_metadata(hparams)
  # 执行学生模型的训练计划
  t2t_trainer.execute_schedule(exp)
  # ==========================


def create_teacher_experiment(run_config, hparams, argv):
  """Creates experiment function."""
  tf.logging.info("training teacher")
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    return cloud_mlengine.launch()

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  hparams.distill_phase = "train"
  exp_fn = t2t_trainer.create_experiment_fn()
  exp = exp_fn(run_config, hparams)
  return exp


def create_student_experiment(run_config, hparams, argv):
  """Creates experiment function."""
  tf.logging.info("training student")
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    return cloud_mlengine.launch()

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  hparams.add_hparam("teacher_dir", FLAGS.teacher_dir)
  hparams.add_hparam("student_dir", FLAGS.student_dir)
  hparams.distill_phase = "distill"
  exp_fn = t2t_trainer.create_experiment_fn()
  exp = exp_fn(run_config, hparams)
  return exp


def create_experiment_fn(argv, train_teacher):

  def teacher_experiment_fn(run_config, hparams):
    return create_teacher_experiment(run_config, hparams, argv)

  def student_experiment_fn(run_config, hparams):
    return create_student_experiment(run_config, hparams, argv)

  return teacher_experiment_fn if train_teacher else student_experiment_fn


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
