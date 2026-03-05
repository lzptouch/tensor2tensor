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

"""Transformer 的记忆单元。

实现用于 Transformer 模型的循环记忆机制。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf


class RecurrentMemory(object):
  """循环记忆的基类。

  该类定义了记忆接口，但行为类似于空操作。
  """

  def pre_attention(self, segment, query_antecedent, memory_antecedent, bias):
    """在自注意力之前调用，用于整合记忆项。

    参数：
        segment: 形状为 [batch] 的整数 Tensor
        query_antecedent: 形状为 [batch, length_q, channels] 的 Tensor
        memory_antecedent: 必须为 None。注意力通常允许这是一个
            形状为 [batch, length_m, channels] 的 Tensor，但目前我们只
            支持解码器侧自注意力的记忆。
        bias: 偏置 Tensor（参见 attention_bias()）

    返回：
        (data, new_query_antecedent, new_memory_antecedent, new_bias)
    """
    del segment
    return None, query_antecedent, memory_antecedent, bias

  def post_attention(self, token, x):
    """在自注意力之后调用。记忆可以在此更新。

    参数：
        token: pre_attention 返回的数据，可用于传递与当前记忆操作相关的状态
        x: 自注意力和前馈之后的数据 Tensor

    返回：
        输入 x 的（可能修改过的）版本
    """
    assert token is None
    return x


class RecentTokensMemory(RecurrentMemory):
  """缓存最近标记特征的记忆模块。

  当缓存的标记数量等于块大小时，这等价于 Transformer-XL
  （https://arxiv.org/abs/1901.02860）中使用的记忆。
  """

  def __init__(self, name, hparams):
    """初始化最近标记记忆模块。

    参数：
        name: 模块名称
        hparams: 超参数
    """
    hidden_size = hparams.hidden_size
    self.chunk_length = hparams.split_targets_chunk_length
    assert self.chunk_length > 0, "Chunking is required to use recurrent memory"

    if hasattr(hparams, "num_memory_items") and hparams.num_memory_items > 0:
      self.tokens_to_cache = hparams.num_memory_items
    else:
      self.tokens_to_cache = self.chunk_length

    # TODO(kitaev): The implementation of the chunking code makes it somewhat
    # convoluted to figure out how many actual sequences we can have per batch.
    # The data pipeline should be revisited at some point.
    if (hasattr(hparams, "recurrent_memory_batch_size")
        and hparams.recurrent_memory_batch_size > 0):
      batch_size_in_sequences = hparams.recurrent_memory_batch_size
    else:
      batch_size_in_sequences = hparams.batch_size / hparams.max_length

    memory_shape = [batch_size_in_sequences, self.tokens_to_cache, hidden_size]
    bias_shape = [batch_size_in_sequences, 1, 1, self.tokens_to_cache]

    with tf.variable_scope(name):
      self.previous_segment = tf.get_variable(
          "memsegment", (batch_size_in_sequences,),
          dtype=tf.int32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          initializer=tf.constant_initializer(0))

      self.previous_vals = tf.get_variable(
          "memvals", memory_shape,
          dtype=tf.float32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          initializer=tf.constant_initializer(.0))

      self.previous_bias = tf.get_variable(
          "membias", bias_shape,
          dtype=tf.float32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          initializer=tf.constant_initializer(-1e9))

  def pre_attention(self, segment, query_antecedent, memory_antecedent, bias):
    """Called prior to self-attention, to incorporate memory items.

    Args:
      segment: an integer Tensor with shape [batch]
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: must be None. Attention normally allows this to be a
        Tensor with shape [batch, length_m, channels], but we currently only
        support memory for decoder-side self-attention.
      bias: bias Tensor (see attention_bias())
    Returns:
      (data, new_query_antecedent, new_memory_antecedent, new_bias)
    """
    assert memory_antecedent is None, "We only support language modeling"

    # In eval mode, batch size may be variable
    memory_batch_size = tf.shape(self.previous_vals)[0]
    current_batch_size = tf.shape(query_antecedent)[0]
    amount_to_pad = memory_batch_size - current_batch_size

    # If segment id is zero, don't attend back to the memory
    previous_bias = self.previous_bias[:current_batch_size, :, :, :] + tf.cast(
        tf.equal(segment[:, None, None, None], 0), tf.float32) * -1e9

    sliced_previous_vals = self.previous_vals[:current_batch_size, :, :]

    new_memory_antecedent = tf.concat(
        [tf.stop_gradient(sliced_previous_vals), query_antecedent], 1)
    new_bias = tf.concat([
        tf.tile(tf.stop_gradient(previous_bias), [1, 1, self.chunk_length, 1]),
        tf.tile(bias, [current_batch_size, 1, 1, 1]),
    ], -1)

    remember_segment = tf.pad(segment, [[0, amount_to_pad]])
    # TODO(kitaev): The code assumes that we always either increment the chunk
    # number or reset it to zero. This assumption will not hold if we re-run the
    # model for each token, e.g. for autoregressive greedy/beam/sampling decode.
    remember_vals = tf.pad(query_antecedent,
                           [[0, amount_to_pad], [0, 0], [0, 0]])
    # Query position is on axis -2 for bias: as long as a token can be attended
    # to from at least one query position (i.e. it's not padding), memorize it.
    remember_bias = tf.tile(
        tf.reduce_max(bias, -2, keepdims=True), [memory_batch_size, 1, 1, 1])
    # Assume that query_antecedent is always a full chunk (i.e. not truncated)
    if self.chunk_length < self.tokens_to_cache:
      remember_vals = tf.concat([self.previous_vals, remember_vals], 1)
      remember_bias = tf.concat([
          self.previous_bias - 1e9 * tf.cast(
              tf.equal(
                  tf.pad(segment, [[0, amount_to_pad]])[:, None, None, None],
                  0), tf.float32),
          remember_bias
      ], -1)
    if self.chunk_length != self.tokens_to_cache:
      remember_vals = remember_vals[:, -self.tokens_to_cache:, :]
      remember_bias = remember_bias[:, :, :, -self.tokens_to_cache:]
    token = (remember_segment, remember_vals, remember_bias)

    return token, query_antecedent, new_memory_antecedent, new_bias

  def post_attention(self, token, x):
    """Called after self-attention. The memory can be updated here.

    Args:
      token: Data returned by pre_attention, which can be used to carry over
        state related to the current memory operation.
      x: a Tensor of data after self-attention and feed-forward
    Returns:
      a (possibly modified) version of the input x
    """
    with tf.control_dependencies([
        self.previous_segment.assign(token[0]),
        self.previous_vals.assign(token[1]),
        self.previous_bias.assign(token[2]),
        ]):
      return tf.identity(x)


class TransformerMemory(object):
  """Implements the Memory module.

  Based on Neural Turing Machines: arXiv:1410.5401 [cs.NE]
  """

  def __init__(self, batch_size, key_depth, val_depth, memory_size,
               sharpen_factor=1., name="neural_memory"):
    """Initialize the memory object.

    Args:
      batch_size: the batch size.
      key_depth: the depth of the memory keys.
      val_depth: the depth of the memory values.
      memory_size: the number of items in the memory.
      sharpen_factor: the sharpen_factor for addressing the memory.
      name: the optional variable scope.
    """
    self.name = name
    self.batch_size = batch_size
    self.key_depth = key_depth
    self.val_depth = val_depth
    self.memory_size = memory_size
    self.sharpen_factor = sharpen_factor
    with tf.variable_scope(name):
      self.segment_number = tf.get_variable(
          "segment_number", [self.batch_size],
          dtype=tf.int32, trainable=False,
          initializer=tf.constant_initializer(100000))
      self.mem_vals = tf.get_variable(
          "memvals", [self.batch_size, self.memory_size, self.val_depth],
          dtype=tf.float32, trainable=False,
          initializer=tf.constant_initializer(.0))
      self.mean_logits = tf.get_variable(
          "meanlogits", [self.batch_size, self.memory_size],
          dtype=tf.float32, trainable=False,
          initializer=tf.constant_initializer(.0))

  def _norm(self, x):
    """Compute the safe norm."""
    return tf.sqrt(tf.reduce_sum(tf.square(x), keepdims=True, axis=-1) + 1e-7)

  def _address_content(self, x):
    """Address the memory based on content similarity.

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
    Returns:
      the logits for each memory entry [batch_size, length, memory_size].
    """
    mem_keys = tf.layers.dense(self.mem_vals, self.key_depth,
                               bias_initializer=tf.constant_initializer(1.0),
                               name="mem_key")
    mem_query = tf.layers.dense(x, self.key_depth,
                                bias_initializer=tf.constant_initializer(1.0),
                                name="mem_query")
    norm = tf.matmul(self._norm(mem_query), self._norm(mem_keys),
                     transpose_b=True)
    dot_product = tf.matmul(mem_query, mem_keys, transpose_b=True)
    cos_dist = tf.div(dot_product, norm + 1e-7, name="cos_dist")
    access_logits = self.sharpen_factor * cos_dist
    return access_logits

  def read(self, x):
    """Read from the memory.

    An external component can use the results via a simple MLP,
    e.g., fn(x W_x + retrieved_mem W_m).

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
    Returns:
      access_logits: the logits for accessing the memory in shape of
          [batch_size, length, memory_size].
      retrieved_mem: the retrieved results in the shape of
          [batch_size, length, val_depth].
    """
    access_logits = self._address_content(x)
    weights = tf.nn.softmax(access_logits)
    retrieved_mem = tf.reduce_sum(
        tf.multiply(tf.expand_dims(weights, 3),
                    tf.expand_dims(self.mem_vals, axis=1)), axis=2)
    return access_logits, retrieved_mem

  def write(self, x, access_logits):
    """Write to the memory based on a combination of similarity and least used.

    Based on arXiv:1607.00036v2 [cs.LG].

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
      access_logits: the logits for accessing the memory.
    Returns:
      the update op.
    """
    gamma = tf.layers.dense(x, 1, activation=tf.sigmoid, name="gamma")
    write_logits = access_logits - gamma * tf.expand_dims(self.mean_logits, 1)
    candidate_value = tf.layers.dense(x, self.val_depth,
                                      activation=tf.nn.relu,
                                      name="candidate_value")
    erase_gates = tf.layers.dense(x, self.memory_size,
                                  activation=tf.nn.sigmoid,
                                  name="erase")
    write_weights = tf.nn.softmax(write_logits)
    erase_weights = tf.expand_dims(1 - erase_gates * write_weights, 3)
    erase = tf.multiply(erase_weights,
                        tf.expand_dims(self.mem_vals, 1))
    addition = tf.multiply(
        tf.expand_dims(write_weights, 3),
        tf.expand_dims(candidate_value, 2))
    update_value_op = self.mem_vals.assign(
        tf.reduce_mean(erase + addition, axis=1))
    with tf.control_dependencies([update_value_op]):
      write_op = self.mean_logits.assign(
          self.mean_logits * 0.1 + tf.reduce_mean(write_logits * 0.9, axis=1))
      return write_op

  def set(self, mem_vals, mean_logits):
    set_op = tf.group([
        self.mem_vals.assign(mem_vals),
        self.mean_logits.assign(mean_logits)])
    return set_op

  def get(self):
    return self.mem_vals, self.mean_logits

  def update_segment_number(self, segment_number):
    return self.segment_number.assign(segment_number)

  def reset(self, entries_to_reset):
    """Reset the entries in the memory.

    Args:
      entries_to_reset: a 1D tensor.
    Returns:
      the reset op.
    """
    num_updates = tf.size(entries_to_reset)
    update_vals = tf.scatter_update(
        self.mem_vals, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size, self.val_depth], .0), 0),
                [num_updates, 1, 1]))
    update_logits = tf.scatter_update(
        self.mean_logits, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size], .0), 0),
                [num_updates, 1]))
    reset_op = tf.group([update_vals, update_logits])
    return reset_op

  def pre_attention(self, segment_number, query_antecedent,
                    memory_antecedent, bias):
    """Called prior to self-attention, to incorporate memory items.

    Args:
      segment_number: an integer Tensor with shape [batch]
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: must be None. Attention normally allows this to be a
        Tensor with shape [batch, length_m, channels], but we currently only
        support memory for decoder-side self-attention.
      bias: bias Tensor (see attention_bias())
    Returns:
      (data, new_query_antecedent, new_memory_antecedent, new_bias)
    """
    with tf.variable_scope(self.name + "/pre_attention", reuse=tf.AUTO_REUSE):
      assert memory_antecedent is None, "We only support language modeling"
      with tf.control_dependencies([
          tf.assert_greater_equal(self.batch_size, tf.size(segment_number))]):
        difference = self.batch_size - tf.size(segment_number)
        segment_number = tf.pad(segment_number, [[0, difference]])
        reset_op = self.reset(tf.reshape(tf.where(
            tf.less(segment_number, self.segment_number)), [-1]))
      memory_results = {}
      with tf.control_dependencies([reset_op]):
        with tf.control_dependencies([
            self.update_segment_number(segment_number)]):
          x = tf.pad(query_antecedent, [
              [0, difference], [0, 0], [0, 0]])
          access_logits, retrieved_mem = self.read(x)
      memory_results["x"] = x
      memory_results["access_logits"] = access_logits
      memory_results["retrieved_mem"] = retrieved_mem
      return memory_results, query_antecedent, memory_antecedent, bias

  def post_attention(self, token, x):
    """Called after self-attention. The memory can be updated here.

    Args:
      token: Data returned by pre_attention, which can be used to carry over
        state related to the current memory operation.
      x: a Tensor of data after self-attention and feed-forward
    Returns:
      a (possibly modified) version of the input x
    """
    with tf.variable_scope(self.name + "/post_attention", reuse=tf.AUTO_REUSE):
      depth = common_layers.shape_list(x)[-1]
      actual_batch_size = common_layers.shape_list(x)[0]
      memory_output = tf.gather(token["retrieved_mem"],
                                tf.range(actual_batch_size))
      output = tf.add(tf.layers.dense(x, depth, use_bias=False),
                      tf.layers.dense(memory_output, depth))
      with tf.control_dependencies([output]):
        with tf.control_dependencies([
            self.write(token["x"], token["access_logits"])]):
          return tf.identity(output)
