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

"""RNN LSTM 模型。

实现基于长短期记忆网络（LSTM）的循环神经网络模型，
用于序列建模任务。

功能说明：
- 实现 LSTM（Long Short-Term Memory）网络
- 支持多层 LSTM 堆叠
- 支持 Dropout 正则化
- 适用于序列标注、语言建模、机器翻译等任务
- 提供编码器 - 解码器架构支持
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from tensor2tensor.layers import area_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


def _dropout_lstm_cell(hparams, train):
  """创建带 Dropout 的 LSTM 单元。
  
  Args:
    hparams: 超参数对象，包含 hidden_size、dropout 等配置
    train: 布尔值，指示是否处于训练模式
  
  Returns:
    带有 Dropout 包装的 LSTMCell
  
  功能说明：
  - 创建 LSTM 细胞单元
  - 应用 Dropout 防止过拟合
  - 仅在训练模式时启用 Dropout
  """
  return tf.nn.rnn_cell.DropoutWrapper(
      tf.nn.rnn_cell.LSTMCell(hparams.hidden_size),
      input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


def lstm(inputs, sequence_length, hparams, train, name, initial_state=None):
  """在输入顶部添加 LSTM 层堆栈。

  参数：
      inputs: 输入张量，形状为 `[batch_size, time_steps, hidden_size]`
      sequence_length: 实际输入序列的长度（不包括填充），形状为 `[batch_size]`
      hparams: 超参数对象
      train: 布尔值，构建训练图时启用 dropout 为 True
      name: 字符串，在此作用域下创建变量名
      initial_state: LSTMStateTuple 元组，每层的初始状态

  返回：
      元组 (outputs, states)，其中：
          outputs: 输出张量，形状为 `[batch_size, time_steps, hidden_size]`
          states: LSTMStateTuple 元组，每层的最终状态
              双向 LSTM 返回最后一个前向和后向状态的连接，
              减少到原始维度
  """
  layers = [_dropout_lstm_cell(hparams, train)
            for _ in range(hparams.num_hidden_layers)]
  with tf.variable_scope(name):
    return tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        inputs,
        sequence_length,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)


def lstm_attention_decoder(inputs, hparams, train, name, initial_state,
                           encoder_outputs, encoder_output_length,
                           decoder_input_length):
  """在形状为 [batch x time x size] 的输入上运行带注意力的 LSTM 单元。

  参数：
      inputs: 解码器输入张量，形状为 `[batch_size, decoder_steps, hidden_size]`
      hparams: 超参数对象
      train: 布尔值，构建训练图时启用 dropout 为 True
      name: 字符串，在此作用域下创建变量名
      initial_state: LSTMStateTuple 元组，每层的初始状态
      encoder_outputs: 编码器输出，形状为 `[batch_size, encoder_steps, hidden_size]`
      encoder_output_length: 实际编码器输出的长度（不包括填充），形状为 `[batch_size]`
      decoder_input_length: 实际解码器输入的长度（不包括填充），形状为 `[batch_size]`

  异常：
      ValueError: 如果 hparams.attention_mechanism 不是 luong 或 bahdanau

  返回：
      解码器输出张量，形状为 `[batch_size, decoder_steps, hidden_size]`
  """
  layers = [_dropout_lstm_cell(hparams, train)
            for _ in range(hparams.num_hidden_layers)]
  if hparams.attention_mechanism == "luong":
    attention_mechanism_class = contrib.seq2seq().LuongAttention
  elif hparams.attention_mechanism == "bahdanau":
    attention_mechanism_class = contrib.seq2seq().BahdanauAttention
  else:
    raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                     "luong or bahdanau." % hparams.attention_mechanism)
  if hparams.get("max_area_width", 1) > 1:
    def _area_key_value_fn(keys, values):
      """Custom fn for computing area keys and values."""
      tf.logging.info("max_area_width=%d, area_key_mode=%s, area_value_mode=%s",
                      hparams.get("max_area_width", 1),
                      hparams.get("area_key_mode", "none"),
                      hparams.get("area_value_mode", "none"))
      keys = area_attention.compute_area_key(
          keys, max_area_width=hparams.get("max_area_width", 1),
          mode=hparams.get("area_key_mode", "none"), name="decoder_encoder",
          training=(hparams.mode == tf_estimator.ModeKeys.TRAIN))
      if hparams.get("area_value_mode", "none") == "sum":
        _, _, values, _, _ = area_attention.compute_area_features(
            values, max_area_width=hparams.get("max_area_width", 1))
      elif hparams.get("area_value_mode", "none") == "mean":
        values, _, _, _, _ = area_attention.compute_area_features(
            values, max_area_width=hparams.get("max_area_width", 1))
      else:
        raise ValueError(
            "Unsupported area_value_mode: %s" % hparams.get(
                "area_value_mode", "none"))
      return keys, values
    area_mask = area_attention.lengths_to_area_mask(
        feature_length=encoder_output_length,
        length=common_layers.shape_list(encoder_outputs)[1],
        max_area_size=hparams.get("max_area_width", "1"))
    def _area_prob_fn(score):
      alignments = tf.nn.softmax(score)
      alignments = tf.where(area_mask, alignments, tf.zeros_like(alignments))
      alignments = tf.div(alignments, tf.reduce_sum(
          alignments, axis=-1, keepdims=True))
      return alignments
    attention_mechanism = attention_mechanism_class(
        hparams.hidden_size, encoder_outputs,
        memory_sequence_length=None,
        probability_fn=_area_prob_fn,
        custom_key_value_fn=_area_key_value_fn)
  else:
    attention_mechanism = attention_mechanism_class(hparams.hidden_size,
                                                    encoder_outputs)
  cell = contrib.seq2seq().AttentionWrapper(
      tf.nn.rnn_cell.MultiRNNCell(layers),
      [attention_mechanism] * hparams.num_heads,
      attention_layer_size=[hparams.attention_layer_size] * hparams.num_heads,
      output_attention=(hparams.output_attention == 1))

  batch_size = common_layers.shape_list(inputs)[0]

  initial_state = cell.zero_state(batch_size, tf.float32).clone(
      cell_state=initial_state)

  with tf.variable_scope(name):
    output, _ = tf.nn.dynamic_rnn(
        cell,
        inputs,
        decoder_input_length,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)
    # output is [batch_size, decoder_steps, attention_size], where
    # attention_size is either hparams.hidden_size (when
    # hparams.output_attention is 0) or hparams.attention_layer_size (when
    # hparams.output_attention is 1) times the number of attention heads.
    #
    # For multi-head attention project output back to hidden size.
    if hparams.output_attention == 1 and hparams.num_heads > 1:
      output = tf.layers.dense(output, hparams.hidden_size)

    return output


def lstm_seq2seq_internal(inputs, targets, hparams, train):
  """The basic LSTM seq2seq model, main step used for training."""
  with tf.variable_scope("lstm_seq2seq"):
    if inputs is not None:
      inputs_length = common_layers.length_from_embedding(inputs)
      # Flatten inputs.
      inputs = common_layers.flatten4d3d(inputs)

      # LSTM encoder.
      inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
      _, final_encoder_state = lstm(inputs, inputs_length, hparams, train,
                                    "encoder")
    else:
      final_encoder_state = None

    # LSTM decoder.
    shifted_targets = common_layers.shift_right(targets)
    # Add 1 to account for the padding added to the left from shift_right
    targets_length = common_layers.length_from_embedding(shifted_targets) + 1
    decoder_outputs, _ = lstm(
        common_layers.flatten4d3d(shifted_targets),
        targets_length,
        hparams,
        train,
        "decoder",
        initial_state=final_encoder_state)
    return tf.expand_dims(decoder_outputs, axis=2)


def lstm_seq2seq_internal_attention(inputs, targets, hparams, train,
                                    inputs_length, targets_length):
  """LSTM seq2seq model with attention, main step used for training."""
  with tf.variable_scope("lstm_seq2seq_attention"):
    # Flatten inputs.
    inputs = common_layers.flatten4d3d(inputs)

    # LSTM encoder.
    inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
    encoder_outputs, final_encoder_state = lstm(
        inputs, inputs_length, hparams, train, "encoder")

    # LSTM decoder with attention.
    shifted_targets = common_layers.shift_right(targets)
    # Add 1 to account for the padding added to the left from shift_right
    targets_length = targets_length + 1
    decoder_outputs = lstm_attention_decoder(
        common_layers.flatten4d3d(shifted_targets), hparams, train, "decoder",
        final_encoder_state, encoder_outputs, inputs_length, targets_length)
    return tf.expand_dims(decoder_outputs, axis=2)


def lstm_bid_encoder(inputs, sequence_length, hparams, train, name):
  """Bidirectional LSTM for encoding inputs that are [batch x time x size]."""

  with tf.variable_scope(name):
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [_dropout_lstm_cell(hparams, train)
         for _ in range(hparams.num_hidden_layers)])

    cell_bw = tf.nn.rnn_cell.MultiRNNCell(
        [_dropout_lstm_cell(hparams, train)
         for _ in range(hparams.num_hidden_layers)])

    ((encoder_fw_outputs, encoder_bw_outputs),
     (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
         cell_fw,
         cell_bw,
         inputs,
         sequence_length,
         dtype=tf.float32,
         time_major=False)

    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    encoder_states = []

    for i in range(hparams.num_hidden_layers):
      if isinstance(encoder_fw_state[i], tf.nn.rnn_cell.LSTMStateTuple):
        encoder_state_c = tf.concat(
            values=(encoder_fw_state[i].c, encoder_bw_state[i].c),
            axis=1,
            name="encoder_fw_state_c")
        encoder_state_h = tf.concat(
            values=(encoder_fw_state[i].h, encoder_bw_state[i].h),
            axis=1,
            name="encoder_fw_state_h")
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_state_c, h=encoder_state_h)
      elif isinstance(encoder_fw_state[i], tf.Tensor):
        encoder_state = tf.concat(
            values=(encoder_fw_state[i], encoder_bw_state[i]),
            axis=1,
            name="bidirectional_concat")

      encoder_states.append(encoder_state)

    encoder_states = tuple(encoder_states)
    return encoder_outputs, encoder_states


def lstm_seq2seq_internal_bid_encoder(inputs, targets, hparams, train):
  """The basic LSTM seq2seq model with bidirectional encoder."""
  with tf.variable_scope("lstm_seq2seq_bid_encoder"):
    if inputs is not None:
      inputs_length = common_layers.length_from_embedding(inputs)
      # Flatten inputs.
      inputs = common_layers.flatten4d3d(inputs)
      # LSTM encoder.
      _, final_encoder_state = lstm_bid_encoder(
          inputs, inputs_length, hparams, train, "encoder")
    else:
      inputs_length = None
      final_encoder_state = None
    # LSTM decoder.
    shifted_targets = common_layers.shift_right(targets)
    # Add 1 to account for the padding added to the left from shift_right
    targets_length = common_layers.length_from_embedding(shifted_targets) + 1
    hparams_decoder = copy.copy(hparams)
    hparams_decoder.hidden_size = 2 * hparams.hidden_size
    decoder_outputs, _ = lstm(
        common_layers.flatten4d3d(shifted_targets),
        targets_length,
        hparams_decoder,
        train,
        "decoder",
        initial_state=final_encoder_state)
    return tf.expand_dims(decoder_outputs, axis=2)


def lstm_seq2seq_internal_attention_bid_encoder(inputs, targets, hparams,
                                                train):
  """LSTM seq2seq model with attention, main step used for training."""
  with tf.variable_scope("lstm_seq2seq_attention_bid_encoder"):
    inputs_length = common_layers.length_from_embedding(inputs)
    # Flatten inputs.
    inputs = common_layers.flatten4d3d(inputs)
    # LSTM encoder.
    encoder_outputs, final_encoder_state = lstm_bid_encoder(
        inputs, inputs_length, hparams, train, "encoder")
    # LSTM decoder with attention
    shifted_targets = common_layers.shift_right(targets)
    # Add 1 to account for the padding added to the left from shift_right
    targets_length = common_layers.length_from_embedding(shifted_targets) + 1
    hparams_decoder = copy.copy(hparams)
    hparams_decoder.hidden_size = 2 * hparams.hidden_size
    decoder_outputs = lstm_attention_decoder(
        common_layers.flatten4d3d(shifted_targets), hparams_decoder, train,
        "decoder", final_encoder_state, encoder_outputs,
        inputs_length, targets_length)
    return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class LSTMEncoder(t2t_model.T2TModel):
  """LSTM encoder only."""

  def body(self, features):
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf_estimator.ModeKeys.TRAIN
    inputs = features.get("inputs")
    inputs_length = common_layers.length_from_embedding(inputs)
    # Flatten inputs.
    inputs = common_layers.flatten4d3d(inputs)
    # LSTM encoder.
    inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
    encoder_output, _ = lstm(inputs, inputs_length, self._hparams, train,
                             "encoder")
    return tf.expand_dims(encoder_output, axis=2)


@registry.register_model
class LSTMSeq2seq(t2t_model.T2TModel):

  def body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf_estimator.ModeKeys.TRAIN
    return lstm_seq2seq_internal(features.get("inputs"), features["targets"],
                                 self._hparams, train)


@registry.register_model
class LSTMSeq2seqAttention(t2t_model.T2TModel):
  """Seq to seq LSTM with attention."""

  def body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf_estimator.ModeKeys.TRAIN
    # This is a temporary fix for varying-length sequences within in a batch.
    # A more complete fix should pass a length tensor from outside so that
    # all the lstm variants can use it.
    input_shape = common_layers.shape_list(features["inputs_raw"])
    flat_input = tf.reshape(features["inputs_raw"],
                            [input_shape[0], input_shape[1]])
    inputs_length = tf.reduce_sum(tf.minimum(flat_input, 1), -1)
    target_shape = common_layers.shape_list(features["targets_raw"])
    flat_target = tf.reshape(features["targets_raw"],
                             [target_shape[0], target_shape[1]])
    targets_length = tf.reduce_sum(tf.minimum(flat_target, 1), -1)
    tf.logging.info(self._hparams)
    return lstm_seq2seq_internal_attention(
        features["inputs"], features["targets"], self._hparams, train,
        inputs_length, targets_length)


@registry.register_model
class LSTMSeq2seqBidirectionalEncoder(t2t_model.T2TModel):

  def body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf_estimator.ModeKeys.TRAIN
    return lstm_seq2seq_internal_bid_encoder(
        features.get("inputs"), features["targets"], self._hparams, train)


@registry.register_model
class LSTMSeq2seqAttentionBidirectionalEncoder(t2t_model.T2TModel):

  def body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf_estimator.ModeKeys.TRAIN
    return lstm_seq2seq_internal_attention_bid_encoder(
        features.get("inputs"), features["targets"], self._hparams, train)


@registry.register_hparams
def lstm_seq2seq():
  """hparams for LSTM."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  return hparams


def lstm_attention_base():
  """Base attention params."""
  hparams = lstm_seq2seq()
  hparams.add_hparam("attention_layer_size", hparams.hidden_size)
  hparams.add_hparam("output_attention", True)
  hparams.add_hparam("num_heads", 1)
  return hparams


@registry.register_hparams
def lstm_bahdanau_attention():
  """Hparams for LSTM with bahdanau attention."""
  hparams = lstm_attention_base()
  hparams.add_hparam("attention_mechanism", "bahdanau")
  return hparams


@registry.register_hparams
def lstm_luong_attention():
  """Hparams for LSTM with luong attention."""
  hparams = lstm_attention_base()
  hparams.add_hparam("attention_mechanism", "luong")
  return hparams


@registry.register_hparams
def lstm_attention():
  """For backwards compatibility, defaults to bahdanau."""
  return lstm_bahdanau_attention()


@registry.register_hparams
def lstm_bahdanau_attention_multi():
  """Multi-head Bahdanau attention."""
  hparams = lstm_bahdanau_attention()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def lstm_luong_attention_multi():
  """Multi-head Luong attention."""
  hparams = lstm_luong_attention()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def lstm_asr_v1():
  """Basic LSTM Params."""
  hparams = lstm_bahdanau_attention()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.batch_size = 36
  hparams.max_input_seq_length = 600000
  hparams.max_target_seq_length = 350
  hparams.max_length = hparams.max_input_seq_length
  hparams.min_length_bucket = hparams.max_input_seq_length // 2
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def lstm_area_attention_base():
  """Hparams for LSTM with area attention."""
  hparams = lstm_luong_attention()
  hparams.batch_size = 16384
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 1024
  hparams.num_heads = 4
  hparams.dropout = 0.2
  hparams.learning_rate = 0.1
  hparams.max_area_width = 2
  hparams.area_key_mode = "mean"
  hparams.area_value_mode = "sum"
  return hparams


@registry.register_hparams
def lstm_area_attention_enfr():
  """Hparams for LSTM with area attention."""
  hparams = lstm_area_attention_base()
  hparams.dropout = 0.1
  return hparams


@registry.register_hparams
def lstm_area_attention_char():
  """Hparams for LSTM with area attention."""
  hparams = lstm_area_attention_base()
  hparams.batch_size = 20480
  return hparams


@registry.register_hparams
def lstm_area_attention_char_enfr():
  """Hparams for LSTM with area attention."""
  hparams = lstm_area_attention_char()
  hparams.dropout = 0.1
  return hparams
