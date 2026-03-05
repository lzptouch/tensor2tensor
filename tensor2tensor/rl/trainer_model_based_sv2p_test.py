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

"""使用 SV2P 随机模型运行 trainer_model_based。冒烟测试。

用于测试基于模型的 RL 训练器的基本功能，特别是 SV2P（随机视频到视频预测）模型。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.rl import trainer_model_based

import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


class ModelRLExperimentSv2pTest(tf.test.TestCase):

  def test_sv2p(self):
    FLAGS.output_dir = tf.test.get_temp_dir()
    FLAGS.loop_hparams_set = "rlmb_tiny_sv2p"
    trainer_model_based.main(None)


if __name__ == "__main__":
  tf.test.main()
