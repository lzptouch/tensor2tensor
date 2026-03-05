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

"""t2t_trainer 的单元测试。

功能说明：
- 测试 t2t_trainer 的基本训练流程
- 验证模型训练、评估功能
- 使用小型数据集进行快速测试
- 继承自 trainer_lib_test 的测试框架
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib_test

import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


class TrainerTest(tf.test.TestCase):
  """训练器测试类。
  
  功能说明：
  - 继承自 TensorFlow 的测试框架
  - 提供端到端的训练测试
  - 验证 t2t_trainer 的完整流程
  """

  @classmethod
  def setUpClass(cls):
    """设置测试类（调用父类的 setUpClass）。
    
    功能说明：
    - 初始化测试环境
    - 准备测试所需的数据和配置
    """
    trainer_lib_test.TrainerLibTest.setUpClass()

  def testTrain(self):
    """测试训练流程。
    
    功能说明：
    - 使用 tiny_algo 问题（小型算法数据集）
    - 使用 transformer 模型
    - 训练 1 步进行基本功能验证
    - 评估 1 步验证评估功能
    """
    # 设置测试参数
    FLAGS.problem = "tiny_algo"  # 使用小型算法问题
    FLAGS.model = "transformer"  # 使用 Transformer 模型
    FLAGS.hparams_set = "transformer_tiny"  # 使用微型超参数配置
    FLAGS.train_steps = 1  # 仅训练 1 步（快速测试）
    FLAGS.eval_steps = 1  # 仅评估 1 步
    FLAGS.output_dir = tf.test.get_temp_dir()  # 使用临时目录作为输出
    FLAGS.data_dir = tf.test.get_temp_dir()  # 使用临时目录作为数据目录
    
    # 执行训练主流程
    t2t_trainer.main(None)


if __name__ == "__main__":
  tf.test.main()
