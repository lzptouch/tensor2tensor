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

"""语言可接受性语料库（CoLA）的数据生成器。

包含用于处理语言可接受性分类任务的函数和类。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

EOS = text_encoder.EOS


@registry.register_problem
class Cola(text_problems.Text2ClassProblem):
  """语言可接受性语料库分类问题。

  用于处理句子是否符合语法规则的二分类任务。
  """

  # 来自 GLUE 基准测试的数据链接：https://gluebenchmark.com/tasks
  _COLA_URL = ("https://firebasestorage.googleapis.com/v0/b/"
               "mtl-sentence-representations.appspot.com/o/"
               "data%2FCoLA.zip?alt=media&token=46d5e637-3411-"
               "4188-bc44-5809b5bfb5f4")

  @property
  def is_generate_per_split(self):
    """是否按分割生成数据。"""
    return True

  @property
  def dataset_splits(self):
    """数据集分割配置。"""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    """近似词汇表大小。"""
    return 2**13  # 8k 词汇表对于这个小数据集已经足够

  @property
  def num_classes(self):
    """类别数量。"""
    return 2

  def class_labels(self, data_dir):
    """返回类别标签列表。

    参数：
        data_dir: 数据目录

    返回：
        类别标签列表
    """

  def _maybe_download_corpora(self, tmp_dir):
    cola_filename = "CoLA.zip"
    cola_finalpath = os.path.join(tmp_dir, "CoLA")
    if not tf.gfile.Exists(cola_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, cola_filename, self._COLA_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return cola_finalpath

  def example_generator(self, filename):
    for line in tf.gfile.Open(filename, "rb"):
      line = text_encoder.to_unicode_utf8(line.strip())
      _, label, _, sent = line.split("\t")
      yield {
          "inputs": sent,
          "label": int(label)
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    cola_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "train.tsv"
    else:
      filesplit = "dev.tsv"

    filename = os.path.join(cola_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class ColaCharacters(Cola):
  """Corpus of Linguistic Acceptability problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.COLA
