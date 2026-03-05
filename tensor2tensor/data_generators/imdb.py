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

"""IMDB 情感分类问题。

包含用于处理 IMDB 电影评论情感分类数据集的类和函数。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class SentimentIMDB(text_problems.Text2ClassProblem):
  """IMDB 情感分类。

  用于处理 IMDB 电影评论的情感二分类问题（正面/负面）。
  """
  URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

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

  def doc_generator(self, imdb_dir, dataset, include_label=False):
    dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
        imdb_dir, dataset, "neg"), False)]

    for d, label in dirs:
      for filename in os.listdir(d):
        with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
          doc = imdb_f.read().strip()
          if include_label:
            yield doc, label
          else:
            yield doc

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    imdb_dir = os.path.join(tmp_dir, "aclImdb")
    if not tf.gfile.Exists(imdb_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate examples
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset = "train" if train else "test"
    for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
      yield {
          "inputs": doc,
          "label": int(label),
      }


@registry.register_problem
class SentimentIMDBCharacters(SentimentIMDB):
  """IMDB sentiment classification, character level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT
