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

"""为 Text2TextProblem 的子类构建词汇表。

构建词汇表的示例用法：

build_vocab \\
    --problem=program_search_algolisp \\
    --data_dir=~/t2t_data \\
    --tmp_dir=~/t2t_data/tmp

功能说明：
- 从注册表中获取指定的问题（problem）
- 为该问题生成词汇表文件
- 词汇表用于将文本转换为模型可以处理的 token ID 序列
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/tmp/t2t/data_dir",
                    "Directory to place the generated vocabulary file in.")

flags.DEFINE_string("tmp_dir", "/tmp/t2t/tmp_dir",
                    "Temporary storage directory.")

flags.DEFINE_string("problem", None,
                    "Problem to generate the vocabulary file for.")

flags.mark_flag_as_required("problem")


def main(_):
  """主函数：执行词汇表构建流程。
  
  Args:
    _: 未使用的参数占位符
  
  流程说明：
  1. 从注册表获取问题实例
  2. 验证问题类型是否为 Text2TextProblem
  3. 创建必要的目录
  4. 生成并保存词汇表
  """
  # 从注册表中获取指定名称的问题实例
  problem = registry.problem(FLAGS.problem)

  # 假设问题是 Text2TextProblem 的子类（用于文本到文本的转换任务）
  assert isinstance(problem, text_problems.Text2TextProblem)

  # 展开用户目录路径（支持~符号）
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)

  # 创建数据目录和临时目录（如果不存在）
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  # 记录日志：保存词汇表到指定目录
  tf.logging.info("Saving vocabulary to data_dir: %s" % data_dir)

  # 获取或创建词汇表（如果已存在则加载，否则生成新的）
  problem.get_or_create_vocab(data_dir, tmp_dir)

  # 记录日志：显示保存的词汇表文件路径
  tf.logging.info("Saved vocabulary file: " +
                  os.path.join(data_dir, problem.vocab_filename))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
