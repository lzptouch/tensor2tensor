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

"""使用 Estimator API 在训练好的 T2T 模型上执行评估。

用于评估训练好的模型性能的工具。

功能说明：
- 在验证集或测试集上评估模型
- 支持多个检查点的连续评估
- 输出评估指标（如 BLEU、accuracy 等）
- 支持超时控制
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer          # pylint: disable=unused-import
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

flags = tf.flags
FLAGS = flags.FLAGS


def main(_):
  """主函数：执行模型评估流程。
  
  Args:
    _: 未使用的参数占位符
  
  流程说明：
  1. 设置随机种子和导入用户模块
  2. 创建超参数对象
  3. 构建评估输入函数
  4. 创建 Estimator 和运行配置
  5. 遍历所有检查点进行评估
  """
  # 设置日志级别为 INFO
  tf.logging.set_verbosity(tf.logging.INFO)
  # 设置随机种子以确保结果可复现
  trainer_lib.set_random_seed(FLAGS.random_seed)
  # 导入用户自定义目录中的模块（如果有）
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # 创建超参数对象
  hparams = trainer_lib.create_hparams(
      FLAGS.hparams_set, FLAGS.hparams, data_dir=FLAGS.data_dir,
      problem_name=FLAGS.problem)

  # 根据 FLAGS.eval_use_test_set 决定使用测试集还是验证集
  dataset_split = "test" if FLAGS.eval_use_test_set else None
  dataset_kwargs = {"dataset_split": dataset_split}
  # 创建评估模式的输入函数
  eval_input_fn = hparams.problem.make_estimator_input_fn(
      tf_estimator.ModeKeys.EVAL, hparams, dataset_kwargs=dataset_kwargs)
  # 创建运行配置（包含 TPU、GPU 等硬件配置）
  config = t2t_trainer.create_run_config(hparams)

  # summary-hook 需要设置 hparams.model_dir
  hparams.add_hparam("model_dir", config.model_dir)

  # 创建 Estimator 用于评估
  estimator = trainer_lib.create_estimator(
      FLAGS.model, hparams, config, use_tpu=FLAGS.use_tpu)
  
  # 获取下一个检查点的迭代器
  ckpt_iter = trainer_lib.next_checkpoint(
      hparams.model_dir, FLAGS.eval_timeout_mins)
  # 遍历所有检查点进行评估
  for ckpt_path in ckpt_iter:
    # 在当前检查点上执行评估
    predictions = estimator.evaluate(
        eval_input_fn, steps=FLAGS.eval_steps, checkpoint_path=ckpt_path)
    # 输出评估结果
    tf.logging.info(predictions)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
