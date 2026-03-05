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

"""ROUGE 指标实现。

这是以下项目的修改和略微扩展版本：
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py

用于评估文本摘要质量的 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）指标。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow.compat.v1 as tf


def _len_lcs(x, y):
  """返回两个序列之间最长公共子序列（LCS）的长度。

  来源：http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  参数：
      x: 单词序列
      y: 单词序列

  返回：
      整数：x 和 y 之间 LCS 的长度
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """计算两个序列之间 LCS 的长度。

  下面的实现使用动态规划算法，时间复杂度为 O(nm)，
  其中 n = len(x)，m = len(y)。

  来源：http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  参数：
      x: 单词集合
      y: 单词集合

  返回：
      包含坐标和 LCS 长度的字典表
  """
  n, m = len(x), len(y)
  table = {}
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _f_lcs(llcs, m, n):
  """计算基于 LCS 的 F-measure 分数。

  来源：https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  参数：
      llcs: LCS 的长度
      m: 参考摘要中的单词数
      n: 候选摘要中的单词数

  返回：
      基于 LCS 的 F-measure 分数（浮点数）
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs


def rouge_l_sentence_level(eval_sentences, ref_sentences):
  """Computes ROUGE-L (sentence level) of two collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set

  Returns:
    A float: F_lcs
  """

  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    m = len(ref_sentence)
    n = len(eval_sentence)
    lcs = _len_lcs(eval_sentence, ref_sentence)
    f1_scores.append(_f_lcs(lcs, m, n))
  return np.mean(f1_scores, dtype=np.float32)


def rouge_l_fscore(predictions, labels, **unused_kwargs):
  """ROUGE scores computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    predictions: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge_l_fscore: approx rouge-l f1 score.
  """
  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(outputs, axis=[-1, -2])
  labels = tf.squeeze(labels, axis=[-1, -2])
  rouge_l_f_score = tf.py_func(rouge_l_sentence_level, (outputs, labels),
                               tf.float32)
  return rouge_l_f_score, tf.constant(1.0)


def _get_ngrams(n, text):
  """Calculates n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
  """Computes ROUGE-N f1 score of two text collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set
    n: Size of ngram.  Defaults to 2.

  Returns:
    f1 score for ROUGE-N
  """

  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    eval_ngrams = _get_ngrams(n, eval_sentence)
    ref_ngrams = _get_ngrams(n, ref_sentence)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if eval_count == 0:
      precision = 0.0
    else:
      precision = overlapping_count / eval_count

    if ref_count == 0:
      recall = 0.0
    else:
      recall = overlapping_count / ref_count

    f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

  # return overlapping_count / reference_count
  return np.mean(f1_scores, dtype=np.float32)


def rouge_2_fscore(predictions, labels, **unused_kwargs):
  """ROUGE-2 F1 score computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    predictions: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge2_fscore: approx rouge-2 f1 score.
  """

  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(outputs, axis=[-1, -2])
  labels = tf.squeeze(labels, axis=[-1, -2])
  rouge_2_f_score = tf.py_func(rouge_n, (outputs, labels), tf.float32)
  return rouge_2_f_score, tf.constant(1.0)
