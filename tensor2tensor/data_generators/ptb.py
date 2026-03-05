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

"""PTB（Penn Treebank）数据集的数据生成器。

包含用于处理 Penn Treebank 语言模型数据集的函数和类。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


EOS = text_encoder.EOS
PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"


def _read_words(filename):
  """从文件中读取单词。

  参数：
      filename: 要读取的文件路径

  返回：
      单词列表
  """
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " %s " % EOS).split()
    else:
      return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()


def _build_vocab(filename, vocab_path, vocab_size):
  """从文件读取并构建包含 `vocab_size` 个最常见单词的词汇表。

  词汇表按出现次数排序，每行一个单词。

  参数：
      filename: 读取单词列表的文件
      vocab_path: 保存词汇表的路径
      vocab_size: 要生成的词汇表大小
  """
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  words = words[:vocab_size]
  with open(vocab_path, "w") as f:
    f.write("\n".join(words))


def _get_token_encoder(vocab_dir, vocab_name, filename):
  """从文件读取并为词汇表返回 `TokenTextEncoder`。

  参数：
      vocab_dir: 词汇表目录
      vocab_name: 词汇表名称
      filename: 文件名

  返回：
      TokenTextEncoder 实例
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    _build_vocab(filename, vocab_path, 10000)
  return text_encoder.TokenTextEncoder(vocab_path)


def _maybe_download_corpus(tmp_dir, vocab_type):
  """下载并解压语料库。

  参数：
      tmp_dir: 包含数据集的目录
      vocab_type: 使用的词汇表类型

  返回：
      文件名列表
  """
  filename = os.path.basename(PTB_URL)
  compressed_filepath = generator_utils.maybe_download(
      tmp_dir, filename, PTB_URL)
  ptb_files = []
  ptb_char_files = []

  with tarfile.open(compressed_filepath, "r:gz") as tgz:
    files = []
    # Selecting only relevant files.
    for m in tgz.getmembers():
      if "ptb" in m.name and ".txt" in m.name:
        if "char" in m.name:
          ptb_char_files += [m.name]
        else:
          ptb_files += [m.name]
        files += [m]

    tgz.extractall(tmp_dir, members=files)

  if vocab_type == text_problems.VocabType.CHARACTER:
    return ptb_char_files
  else:
    return ptb_files


@registry.register_problem
class LanguagemodelPtb10k(text_problems.Text2SelfProblem):
  """PTB, 10k vocab."""

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_filename(self):
    return "vocab.lmptb.10000"

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    files = _maybe_download_corpus(tmp_dir, self.vocab_type)

    train_file, valid_file = None, None
    for filename in files:
      if "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"

    _get_token_encoder(data_dir, self.vocab_filename, train_file)

    train = dataset_split == problem.DatasetSplit.TRAIN
    filepath = train_file if train else valid_file

    def _generate_samples():
      with tf.gfile.GFile(filepath, "r") as f:
        for line in f:
          line = " ".join(line.replace("\n", " %s " % EOS).split())
          yield {"targets": line}

    return _generate_samples()


@registry.register_problem
class LanguagemodelPtbCharacters(LanguagemodelPtb10k):
  """PTB, character-level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
