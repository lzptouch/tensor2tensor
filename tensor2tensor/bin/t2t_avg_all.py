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

"""持续平均给定目录中最后 N 个检查点的脚本。

用于将多个检查点的模型权重进行平均，以提高模型的稳定性和性能。

功能说明：
- 读取指定目录中的多个检查点文件
- 对模型权重进行平均（排除 global_step 和 train_stats）
- 保存平均后的检查点到输出目录
- 支持等待新检查点的连续平均模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import os
import shutil
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.utils import bleu_hook
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "",
                    "Directory to load model checkpoints from.")
flags.DEFINE_string("output_dir", "avg/",
                    "Directory to output the averaged checkpoints to.")
flags.DEFINE_integer("n", 8, "How many checkpoints should be averaged?")
flags.DEFINE_integer("min_steps", 0, "Ignore checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint.")


def main(_):
  """主函数：执行检查点平均流程。
  
  Args:
    _: 未使用的参数占位符
  
  流程说明：
  1. 设置输出目录和文件路径
  2. 复制 flags.txt 以保持时间戳信息
  3. 遍历所有检查点文件
  4. 累加模型权重并计算平均值
  5. 保存平均后的检查点
  """
  # 设置日志级别
  tf.logging.set_verbosity(tf.logging.INFO)

  # 展开用户目录路径
  model_dir = os.path.expanduser(FLAGS.model_dir)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  # 构建输出检查点的基础文件名
  out_base_file = os.path.join(output_dir, "model.ckpt")

  # 复制 flags.txt 文件（如果不存在），以保持原始训练的时间戳信息
  # 这样 t2t-bleu 可以报告正确的相对时间
  tf.gfile.MakeDirs(FLAGS.output_dir)
  if (not os.path.exists(os.path.join(output_dir, "flags.txt")) and
      os.path.exists(os.path.join(model_dir, "flags.txt"))):
    # 使用 copy2 保留文件的元数据（包括时间戳）
    shutil.copy2(os.path.join(model_dir, "flags.txt"),
                 os.path.join(output_dir, "flags.txt"))

  # 初始化已处理模型计数器和队列
  models_processed = 0
  queue = deque()
  
  # 遍历所有检查点文件（按步数排序）
  for model in bleu_hook.stepfiles_iterator(model_dir, FLAGS.wait_minutes,
                                            FLAGS.min_steps):
    # 处理第一个模型时，初始化权重累加器
    if models_processed == 0:
      # 列出检查点中的所有变量
      var_list = tf.train.list_variables(model.filename)
      avg_values = {}
      for (name, shape) in var_list:
        # 跳过 global_step 和 train_stats 变量（这些不需要平均）
        if not (name.startswith("global_step") or
                name.startswith("train_stats/")):
          # 初始化为零数组
          avg_values[name] = np.zeros(shape)
    
    # 增加已处理模型计数
    models_processed += 1

    # 记录日志：显示正在加载的检查点
    tf.logging.info("Loading [%d]: %s" % (models_processed, model.filename))
    # 加载检查点
    reader = tf.train.load_checkpoint(model.filename)
    
    # 累加当前检查点的权重（除以 N 得到平均值的一部分）
    for name in avg_values:
      avg_values[name] += reader.get_tensor(name) / FLAGS.n
    
    # 将当前模型加入队列
    queue.append(model)
    
    # 如果队列中的模型数量不足 N 个，继续等待
    if len(queue) < FLAGS.n:
      continue

    # 构建输出文件名（包含步数信息）
    out_file = "%s-%d" % (out_base_file, model.steps)
    tf_vars = []
    # 记录日志：显示正在平均的检查点
    tf.logging.info("Averaging %s" % (out_file))
    
    # 为每个平均后的权重创建 TensorFlow 变量
    for (name, value) in six.iteritems(avg_values):
      # TODO(martinpopel): dtype=var_dtypes[name]
      tf_vars.append(tf.get_variable(name, shape=value.shape))
    
    # 创建占位符用于传入平均后的权重值
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    # 创建赋值操作：将占位符的值赋给对应的变量
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

    # 创建 global_step 变量，设置为当前模型的步数
    global_step = tf.get_variable(
        "global_step",
        initializer=tf.constant(model.steps, dtype=tf.int64),
        trainable=False)
    
    # 创建 train_stats 作用域（用于兼容其他工具）
    with tf.variable_scope("train_stats"):
      tf.get_variable("problem_0_steps", initializer=0, trainable=False)
    
    # 创建 Saver 用于保存检查点
    saver = tf.train.Saver(tf.global_variables())

    # 记录日志：运行 Session 保存检查点
    tf.logging.info("Running session for %s" % (out_file))
    with tf.Session() as sess:
      # 初始化所有变量
      sess.run(tf.global_variables_initializer())
      # 依次执行赋值操作，将平均后的权重载入模型
      for p, assign_op, (name, value) in zip(
          placeholders, assign_ops, six.iteritems(avg_values)):
        sess.run(assign_op, {p: value})
      
      # 记录日志：保存检查点到文件
      tf.logging.info("Storing to %s" % out_file)
      saver.save(sess, out_base_file, global_step=global_step)
    
    # 设置输出文件的时间戳与源模型一致
    os.utime(out_file + ".index", (model.mtime, model.mtime))

    # 重置 TensorFlow 图，为下一个平均操作做准备
    tf.reset_default_graph()
    
    # 从队列中移除最早的模型
    first_model = queue.popleft()

    # 从平均值中减去最早模型的权重（滑动窗口机制）
    reader = tf.train.load_checkpoint(first_model.filename)
    for name in avg_values:
      avg_values[name] -= reader.get_tensor(name) / FLAGS.n

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
