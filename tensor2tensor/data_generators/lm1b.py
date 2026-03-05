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

"""LM1B 数据集的数据生成器。

包含用于生成 LM1B（10 亿词语言模型）数据集的函数和类。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def _original_vocab(tmp_dir):
  """返回包含原始词汇表的集合。

  这对于与已发表的结果进行比较很重要。

  参数：
      tmp_dir: 包含数据集的目录。

  返回：
      字符串集合
  """
  vocab_url = ("http://download.tensorflow.org/models/LM_LSTM_CNN/"
               "vocab-2016-09-10.txt")
  vocab_filename = os.path.basename(vocab_url + ".en")
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if not os.path.exists(vocab_filepath):
    generator_utils.maybe_download(tmp_dir, vocab_filename, vocab_url)
  return set([
      text_encoder.native_to_unicode(l.strip())
      for l in tf.gfile.Open(vocab_filepath)
  ])


def _replace_oov(original_vocab, line):
  """将词汇表外的词替换为"UNK"。

  这保持了与已发表结果的兼容性。

  参数：
      original_vocab: 字符串集合（数据集的标准词汇表）
      line: Unicode 字符串 - 以空格分隔的词序列。

  返回：
      Unicode 字符串 - 以空格分隔的词序列。
  """
  return u" ".join(
      [word if word in original_vocab else u"UNK" for word in line.split()])


def _train_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "1-billion-word-language-modeling-benchmark-r13output",
                   "training-monolingual.tokenized.shuffled",
                   "news.en-%05d-of-00100" % i) for i in range(1, 100)
  ]


def _dev_data_filenames(tmp_dir):
  return [os.path.join(tmp_dir,
                       "1-billion-word-language-modeling-benchmark-r13output",
                       "heldout-monolingual.tokenized.shuffled",
                       "news.en.heldout-00000-of-00050")]


def _maybe_download_corpus(tmp_dir):
  """下载并解压语料库。

  参数：
      tmp_dir: 包含数据集的目录。
  """
  corpus_url = ("http://www.statmt.org/lm-benchmark/"
                "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
  corpus_filename = os.path.basename(corpus_url)
  corpus_filepath = os.path.join(tmp_dir, corpus_filename)
  if not os.path.exists(corpus_filepath):
    generator_utils.maybe_download(tmp_dir, corpus_filename, corpus_url)
    with tarfile.open(corpus_filepath, "r:gz") as corpus_tar:
      corpus_tar.extractall(tmp_dir)


@registry.register_problem
class LanguagemodelLm1b32k(text_problems.Text2SelfProblem):
  """A language model on the 1B words corpus.

  Ratio of dev tokens (including eos) to dev words (including eos)
  176923 / 159658 = 1.108137; multiply log_ppl by this to compare results.
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def max_samples_for_vocab(self):
    return 63000

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    split_files = {
        problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
        problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
    }
    _maybe_download_corpus(tmp_dir)
    original_vocab = _original_vocab(tmp_dir)
    files = split_files[dataset_split]
    for filepath in files:
      tf.logging.info("filepath = %s", filepath)
      for line in tf.gfile.Open(filepath):
        txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
        yield {"targets": txt}


@registry.register_problem
class LanguagemodelLm1b8k(LanguagemodelLm1b32k):

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192


@registry.register_problem
class LanguagemodelLm1b32kPacked(LanguagemodelLm1b32k):
  """Packed version for TPU training."""

  @property
  def packed_length(self):
    return 256

  @property
  def vocab_filename(self):
    return LanguagemodelLm1b32k().vocab_filename


@registry.register_problem
class LanguagemodelLm1b8kPacked(LanguagemodelLm1b8k):
  """Packed version, 8k vocabulary.

  Ratio of dev tokens (including eos) to dev words (including eos)
  207351 / 159658 = 1.29872; multiply log-ppl by this to compare results.
  """

  @property
  def packed_length(self):
    return 256

  @property
  def vocab_filename(self):
    return LanguagemodelLm1b8k().vocab_filename


@registry.register_problem
class LanguagemodelLm1bCharacters(LanguagemodelLm1b32k):
  """A language model on the 1B words corpus, character level.

  Ratio of dev chars (including eos) to dev words (including eos)
  826189 / 159658 = 5.174742; multiply log-ppl by this to compare results.
  """

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR


@registry.register_problem
class LanguagemodelLm1bCharactersPacked(LanguagemodelLm1bCharacters):
  """Packed version.

  Ratio of dev chars (including eos) to dev words (including eos)
  826189 / 159658 = 5.174742; multiply log-ppl by this to compare results.
  """

  @property
  def packed_length(self):
    return 1024
