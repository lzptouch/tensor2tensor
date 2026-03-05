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

"""计划采样（Scheduled Sampling）。

本模块实现了（Bengio 等人，2015）中描述的计划采样方法。
入口函数有两个：

`sequential_scheduled_sampling_for_t2tmodel()`:
  针对 T2TModel 实例的计划采样方法

`sequential_scheduled_sampling()`:
  计划采样的原始实现，可独立于 T2T 使用

**警告** 此代码非常慢。对于长度为 n 的序列，其运行时至少为 O(n^2)。
对于具有自注意力的模型，其运行时为 O(n^3)。

功能说明：
- 实现 Bengio 等人的计划采样算法
- 用于序列到序列模型的训练
- 逐步从教师强制过渡到自由运行
- 提高模型的泛化能力
- 支持多种采样策略
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import


def sequential_scheduled_sampling_for_t2tmodel(t2tmodel, features):
  """针对 T2TModel 的计划采样。

  参数：
      t2tmodel: T2TModel 实例
      features: {str: Tensor}，输入特征

  返回：
      ss_logits: [batch_size, seq_len, 1, 1, vocab_size]
      losses_dict: {str: scalar Tensor}，需要最小化的损失
  """
  targets = features["targets"]
  targets_size = common_layers.shape_list(targets)
  batch_size = targets_size[0]
  seq_len = targets_size[1]
  targets = tf.reshape(targets, [batch_size, seq_len])

  adapter = ScheduledSamplingAdapter(t2tmodel, features)
  ss_tokens, ss_logits, losses_dict = sequential_scheduled_sampling(
      infer_fn=adapter.infer_fn,
      mix_fn=adapter.mix_fn,
      loss_fn=adapter.loss_fn,
      targets=targets)

  _ = ss_tokens  # unused.
  targets_vocab_size = t2tmodel.problem_hparams.vocab_size["targets"]
  ss_logits = tf.reshape(ss_logits,
                         [batch_size, seq_len, 1, 1, targets_vocab_size])

  return ss_logits, losses_dict


def sequential_scheduled_sampling(infer_fn, mix_fn, loss_fn, targets):
  """计划采样（Scheduled Sampling）。

  参数：
      infer_fn: 函数，计算所有时间步的 logits
      mix_fn: 函数，混合真实标记和采样标记
      loss_fn: 函数，计算真实标记和 logits 之间的损失
      targets: 形状为 [batch_size, seq_len] 的张量，真实标记

  返回：
      ss_tokens: 形状为 [batch_size, seq_len] 的张量，计划采样标记
      ss_logits: 形状为 [batch_size, seq_len, vocab_size] 的张量，
                 基于 ss_tokens 条件预测下一个标记的 logits
      losses_dict: {str: scalar Tensor}，需要优化的损失
  """
  targets_shape = common_layers.shape_list(targets)
  batch_size = targets_shape[0]
  seq_len = targets_shape[1]

  if not targets.shape.is_fully_defined():
    # TODO(duckworthd): When running on GPU, I get the following error. Solve
    # it to enable use on other devices.
    #
    #   Cannot use 'Identity_186' as input to
    #   'transformer/parallel_0_7/transformer/transformer/symbol_modality_16282_512/shared/convert_gradient_to_tensor_HBc3xYw22Mw'
    #   because 'Identity_186' is in a while loop.

    raise ValueError(
        "The following code only works on TPU. As targets.shape isn't fully "
        "defined, I am assuming you are using a different device.")

  def cond_fn(i, ss_tokens):
    """True if i < seq_len."""
    _ = ss_tokens
    return i < seq_len

  def body_fn(i, ss_tokens):
    """Constructs conditioning tokens for scheduled sampling."""
    # next_token_logits depends on timesteps 0...i-1.
    #
    # [batch_size, seq_len] -> [batch_size, seq_len, vocab_size]
    ss_tokens_logits = infer_fn(ss_tokens)

    # Same as 'next_token_logits = ss_tokens_logits[:, i, :]'.
    vocab_size = common_layers.shape_list(ss_tokens_logits)[2]
    next_token_logits = tf.slice(
        ss_tokens_logits, begin=[0, i, 0], size=[batch_size, 1, vocab_size])
    next_token_logits = tf.squeeze(next_token_logits, axis=[1])

    # [batch_size, vocab_size] -> [batch_size]
    sampled_next_tokens = _sample_next_tokens(next_token_logits)

    # Same as 'gold_next_tokens = targets[:, i]'.
    gold_next_tokens = tf.slice(targets, begin=[0, i], size=[batch_size, 1])
    gold_next_tokens = tf.squeeze(gold_next_tokens, axis=[1])

    next_tokens = mix_fn(gold_next_tokens, sampled_next_tokens)
    ss_tokens = _update_timestep(ss_tokens, timestep=i, values=next_tokens)

    return i+1, tf.stop_gradient(ss_tokens)

  # tf.while_loop() over all timesteps. Generate scheduled sampling tokens.
  i = 0
  ss_tokens = tf.zeros([batch_size, seq_len], dtype=tf.int32)
  i, ss_tokens = tf.while_loop(cond_fn, body_fn, [i, ss_tokens])

  ss_logits = infer_fn(ss_tokens)
  return ss_tokens, ss_logits, loss_fn(targets, ss_logits)


def _mix_tokens(p_sample, gold_targets, sampled_targets):
  """Interleave sampled and gold tokens randomly.

  Args:
    p_sample: float in [0, 1]. Probability a token will come from
      'sampled_targets'. 0 means all-gold, 1 means all-sampled.
    gold_targets: Tensor. Gold token IDs.
    sampled_targets: Tensor. Sampled token IDs. Same shape as 'gold_targets'.

  Returns:
    Tensor of same shape as 'gold_targets' containing a mix of tokens from
    'gold_targets' and 'sampled_targets'.
  """
  targets_shape = common_layers.shape_list(sampled_targets)
  return tf.where(
      tf.less(tf.random_uniform(targets_shape), p_sample),
      sampled_targets, gold_targets)


def _sample_next_tokens(logits):
  """Sample tokens for next timestep."""
  batch_size = common_layers.shape_list(logits)[0]
  next_tokens = tf.random.categorical(logits, 1)
  next_tokens = tf.cast(next_tokens, tf.int32)
  next_tokens = tf.reshape(next_tokens, [batch_size])
  return next_tokens


def _update_timestep(x, timestep, values):
  """Set x[:, timestep] = values.

  This operation is **NOT** differentiable.

  Args:
    x: Tensor of shape [batch_size, seq_len, ...]
    timestep: int or scalar Tensor. Index to update in x.
    values: Tensor of shape [batch_size, ...]. New values for x[:, i].

  Returns:
    Copy of 'x' after setting x[:, timestep] = values.
  """
  perm = range(x.shape.ndims)
  perm[0], perm[1] = perm[1], perm[0]
  x = tf.transpose(x, perm)
  x = inplace_ops.alias_inplace_update(x, timestep, values)
  x = tf.transpose(x, perm)
  return x


def inverse_decay_mix_prob(warmup_schedule_name, p_max, num_warmup_steps):
  """Interpolate from 0.001 to 'p_max' over 'num_warmup_steps'."""
  warmup_schedule_fn = {
      "exp": common_layers.inverse_exp_decay,
      "linear": common_layers.inverse_lin_decay,
      "sigmoid": common_layers.inverse_sigmoid_decay,
  }[warmup_schedule_name]
  return p_max * warmup_schedule_fn(num_warmup_steps, min_value=0.001)


class ScheduledSamplingAdapter(object):
  """Adapts T2TModel for sequential_scheduled_sampling()."""

  def __init__(self, t2tmodel, features):
    self._t2tmodel = t2tmodel
    self._features = features

    hparams = self._t2tmodel.hparams
    assert hparams.mode == tf_estimator.ModeKeys.TRAIN, hparams.mode

  def infer_fn(self, partial_targets):
    """Computes logits for all timesteps.

    Args:
      partial_targets: [batch_size, seq_len]. Targets to condition on.

    Returns:
      next_token_logits: [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len = common_layers.shape_list(partial_targets)
    partial_targets = tf.reshape(partial_targets, [batch_size, seq_len, 1, 1])
    features = copy.copy(self._features)
    features["targets"] = partial_targets

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      transformed_features = self._t2tmodel.bottom(features)

      with tf.variable_scope("body"):
        body_outputs, losses = self._t2tmodel._normalize_body_output(  # pylint: disable=protected-access
            self._t2tmodel.body(transformed_features))
        assert losses == {"extra": 0.0}, (
            "Auxiliary losses are not propagated in this code. %s"
            % (losses,))

      logits = self._t2tmodel.top(body_outputs, features)

    vocab_size = self._t2tmodel.problem_hparams.vocab_size["targets"]
    logits = tf.reshape(logits, [batch_size, seq_len, vocab_size])
    return logits

  def mix_fn(self, gold_tokens, sampled_tokens):
    """Mixes gold and sampled tokens randomly."""
    hparams = self._t2tmodel.hparams
    p_sample = inverse_decay_mix_prob(
        hparams.scheduled_sampling_warmup_schedule,
        hparams.scheduled_sampling_gold_mixin_prob,
        hparams.scheduled_sampling_warmup_steps)
    return _mix_tokens(
        p_sample=p_sample,
        gold_targets=gold_tokens,
        sampled_targets=sampled_tokens)

  def loss_fn(self, targets, logits):
    """Constructs loss dict.

    Args:
      targets: [batch_size, seq_len]
      logits: [batch_size, seq_len, vocab_size]

    Returns:
      {str: Tensor of shape []}. Losses.
    """
    batch_size, seq_len, vocab_size = common_layers.shape_list(logits)
    targets = tf.reshape(targets, [batch_size, seq_len, 1, 1])
    logits = tf.reshape(logits, [batch_size, seq_len, 1, 1, vocab_size])
    features = copy.copy(self._features)
    features["targets"] = targets

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      losses = {
          "training": self._t2tmodel.loss(logits, features),
      }

    return losses
