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

"""基于自注意力的语言模型。

类似于 transformer.py，但没有编码器

decoder: [Self-Attention, Feed-forward] x n

该模型结合了混合专家（MoE）层的注意力语言模型，
用于高效的大规模语言建模任务。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


ModeKeys = tf_estimator.ModeKeys  # pylint: disable=invalid-name


class AttentionType(object):
  """注意力层类型的枚举。

  定义了各种注意力层的类型，包括多头注意力、局部专家、
  全局 MoE、内存高效注意力、稀疏注意力等。
  """
  MULTIHEAD = "multihead"
  LOCAL_EXPERTS = "local_experts"
  GLOBAL_MOE = "global_experts"
  MEMORY_EFFICIENT = "memory_efficient"
  SPARSE_MULTIHEAD = "sparse_multihead"
  SPARSE_MULTIHEAD_TRUNCATED = "sparse_multihead_truncated"
  MULTIHEAD_REDUCED = "multihead_reduced"
  MULTIHEAD_FULL = "multihead_full"

  @staticmethod
  def get_choices():
    return [
        AttentionType.MULTIHEAD,
        AttentionType.LOCAL_EXPERTS,
        AttentionType.MEMORY_EFFICIENT,
        AttentionType.SPARSE_MULTIHEAD,
        AttentionType.SPARSE_MULTIHEAD_TRUNCATED,
        AttentionType.MULTIHEAD_REDUCED,
        AttentionType.MULTIHEAD_FULL,
    ]


LAYER_SYMBOLS = {
    "h": AttentionType.MULTIHEAD,  # multi-Head
    "e": AttentionType.LOCAL_EXPERTS,  # Experts
    "m": AttentionType.MEMORY_EFFICIENT,  # Memory
    "s": AttentionType.SPARSE_MULTIHEAD,  # Sparse (Locality sensitive hashing)
    "t": AttentionType.SPARSE_MULTIHEAD_TRUNCATED,  # Using TruncatedDispatcher
    "r": AttentionType.MULTIHEAD_REDUCED,  # Reduced
    "f": AttentionType.MULTIHEAD_FULL,  # Force using full attention
}


@registry.register_model
class AttentionLmMoe(t2t_model.T2TModel):
  """注意力语言模型与混合专家（MoE）。

  结合了自注意力机制和混合专家层的语言模型，
  用于高效的大规模语言建模任务。
  """

  @staticmethod
  def use_body_sharded():
    return True

  def body_sharded(self, sharded_features):
    # Remove dropout if not training
    hparams = self._hparams
    dp = self._data_parallelism
    if hparams.use_inputs:
      decoder_input = dp(tf.squeeze, sharded_features["inputs"], 2)
      decoder_self_attention_bias = None
    else:
      targets = sharded_features["targets"]
      targets = dp(tf.squeeze, targets, 2)
      (decoder_input, decoder_self_attention_bias, pad_remover) = dp(
          attention_lm_moe_prepare_decoder, targets, hparams)

    def preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def postprocess(x, y):
      return dp(common_layers.layer_postprocess, x, y, hparams)

    x = dp(tf.nn.dropout, decoder_input,
           1.0 - hparams.layer_prepostprocess_dropout)
    extra_loss = 0.0

    if not hparams.use_inputs:
      # As preprocess and postprocess are called with batch of size one (all
      # batches concatenated), we just make sure that batch_norm is not use (
      # should not either way)
      assert hparams.norm_type != "batch"

      tf.logging.info("Applying Padding Remover for the attention experts")

      dp_remove_pad = functools.partial(
          dp, remove_pad, pad_remover=pad_remover, mode=hparams.mode)
      dp_restore_pad = functools.partial(
          dp, restore_pad, ref_x=x, pad_remover=pad_remover, mode=hparams.mode)
    else:
      # Using identity function: No effect
      dp_remove_pad = lambda x: x
      dp_restore_pad = lambda x: x

    if hparams.attention_exp_factor != 0:
      tf.logging.info("Expand/compress tokens before sending them to experts")
      dp_expand_bc = lambda x: dp(  # pylint: disable=g-long-lambda
          expand_batch_coordinates,
          x,
          hparams.attention_exp_factor)
      dp_expand_x = lambda x: dp(  # pylint: disable=g-long-lambda
          common_attention.deconv_elems_1d,
          x,
          hparams.attention_exp_factor,
          hparams.attention_exp_inputdim)
      dp_compress_x = lambda x, l: dp(  # pylint: disable=g-long-lambda
          common_attention.conv_elems_1d,
          x,
          hparams.attention_exp_factor,
          l)
    else:
      dp_expand_bc = lambda x: x
      dp_expand_x = lambda x: x
      dp_compress_x = lambda x, l: x

    def print_shape(x, suffix, debug=False):
      # To help debugging, print the input/output shapes at inference and eval
      # Inference for long sequences can take a long time, so that's help to
      # see the progression of the generation
      if not debug and hparams.mode == ModeKeys.TRAIN:
        return x
      return tf.Print(x, [tf.shape(x)], "shape_x_{}".format(suffix))

    with tf.name_scope("batch_coordinate_preprocess"):
      batch_coordinate = dp(get_batch_coordinate, x)
      batch_coordinate = dp_remove_pad(batch_coordinate)
      batch_coordinate = dp_expand_bc(batch_coordinate)
      batch_order = dp(get_batch_coordinate, x, axis=-1)
      batch_order = dp_remove_pad(batch_order)
      batch_order = dp_expand_bc(batch_order)

    x = dp(print_shape, x, "in")

    assert hparams.batch_size >= hparams.max_length

    num_hidden_layers = (
        len(hparams.attention_layers) or hparams.num_hidden_layers)
    for layer in range(num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):

        # Use the layer type defined in attention_layers
        if hparams.attention_layers:
          attention_type = LAYER_SYMBOLS[hparams.attention_layers[layer]]
        else:
          attention_type = hparams.attention_type

        with tf.variable_scope(
            "attention_{}".format(attention_type)):
          if attention_type in [
              AttentionType.MULTIHEAD, AttentionType.MULTIHEAD_FULL]:
            attention_dot_type = (
                "local_mask_right" if hparams.attention_local else
                "dot_product")
            if attention_type == AttentionType.MULTIHEAD_FULL:
              attention_dot_type = "dot_product"
            y = dp(
                common_attention.multihead_attention,
                preprocess(x),
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                attention_type=attention_dot_type,
                block_length=hparams.attention_block_length,
                name="decoder_self_attention")
          elif attention_type == AttentionType.SPARSE_MULTIHEAD:
            x_in = preprocess(x)
            x_in = dp_remove_pad(x_in)
            y, loss_experts = dp(
                common_attention.multihead_attention_sparse_dot_prod,
                x_in,
                None,
                None,  # Bias is computed inside
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,

                # Additional parameters
                bi=[common_attention.BatchInfo(
                    coordinates=batch_coordinate[i],
                    order=batch_order[i],  # No future mask
                ) for i in range(dp.n)],
                use_map_fn=hparams.lsh_use_map_fn,
                experts_params=dict(
                    nb_hyperplanes=hparams.lsh_num_hyperplanes,
                ),
            )
            y = dp_restore_pad(y)

            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss_experts) / dp.n
          elif attention_type == AttentionType.SPARSE_MULTIHEAD_TRUNCATED:
            x_in = preprocess(x)
            y, loss_experts = dp(
                common_attention.multihead_attention_sparse_truncated,
                x_in,
                None,
                None,  # Bias is computed inside
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,

                # Additional parameters
                bi=[common_attention.BatchInfo(
                    coordinates=batch_coordinate[i],
                    order=batch_order[i],  # No future mask
                ) for i in range(dp.n)],
                mask_right=True,
                experts_params=dict(
                    nb_hyperplanes=hparams.lsh_num_hyperplanes,
                ),
            )

            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss_experts) / dp.n
          elif attention_type == AttentionType.MEMORY_EFFICIENT:
            assert hparams.layer_preprocess_sequence == "n"
            y = dp(
                common_attention.multihead_self_attention_memory_efficient,
                x,
                decoder_self_attention_bias,
                hparams.num_heads,
                name="decoder_self_attention")
          elif attention_type == AttentionType.MULTIHEAD_REDUCED:
            y = dp(
                common_attention.multihead_self_attention_reduced,
                preprocess(x),
                factor=hparams.attention_red_factor,
                reduction_type=hparams.attention_reduction_type,
                nonlinearity=hparams.attention_nonlinearity,
                multihead_params=dict(
                    total_key_depth=
                    hparams.attention_key_channels or hparams.hidden_size,
                    total_value_depth=
                    hparams.attention_value_channels or hparams.hidden_size,
                    num_heads=hparams.num_heads,
                    dropout_rate=hparams.attention_dropout,
                ))
          elif attention_type == AttentionType.LOCAL_EXPERTS:
            x_in = preprocess(x)
            x_in = dp_remove_pad(x_in)
            x_in = dp_expand_x(x_in)
            y, loss = dp(
                common_attention.local_expert_attention,
                x_in,
                k=hparams.attention_moe_k,
                loss_coef=hparams.attention_load_balance,
                attention_num_experts=hparams.attention_num_experts,
                train=hparams.mode == ModeKeys.TRAIN,
                batch_coordinate=batch_coordinate,
                mask_right=not hparams.use_inputs,
                split_batch=bool(hparams.attention_split_batch),
                attention_num_head=hparams.attention_num_head,
                attention_kq_size=hparams.attention_kq_size,
                attention_v_size=hparams.attention_v_size)
            y = dp_compress_x(y, x[0].get_shape().as_list()[-1])
            y = dp_restore_pad(y)
            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss) / dp.n
          else:
            raise ValueError("Only {} supported for now.".format(
                AttentionType.get_choices()))
          x = postprocess(x, y)
        with tf.variable_scope("ffn"):
          if hparams.memory_efficient_ffn:
            assert hparams.layer_preprocess_sequence == "n"
            y = dp(
                common_layers.conv_hidden_relu_memory_efficient,
                x,
                hparams.filter_size)
          else:
            additional_conv_params = {}
            if hparams.use_sepconv:
              additional_conv_params = dict(
                  padding="LEFT",
                  # Parameters copied from the transformer model
                  kernel_size=(3, 1),
                  second_kernel_size=(31, 1),
              )
            y = dp(
                common_layers.conv_hidden_relu,
                preprocess(x),
                hparams.filter_size,
                hparams.hidden_size,
                dropout=hparams.relu_dropout,
                **additional_conv_params
            )
          x = postprocess(x, y)
    x = preprocess(x)

    decoder_output = dp(tf.expand_dims, x, 2)
    return decoder_output, extra_loss


def attention_lm_moe_prepare_decoder(targets, hparams):
  """准备解码器的一个分片。

  参数：
      targets: 目标张量
      hparams: 运行超参数

  返回：
      decoder_input: 张量，解码器栈的底部
      decoder_self_attention_bias: 张量，包含大的负值以实现掩码注意力，
          以及可能用于对角对齐的偏置
      pad_remover (expert_utils.PadRemover): 用于移除填充的工具对象
  """
  targets_pad_mask = common_attention.embedding_to_padding(targets)
  with tf.name_scope("pad_remover"):
    # Because of the shift_right, the <eos> token will be considered as
    # padding. In practice, it doesn't really matter, due to the triangular
    # mask, this token should never be attended.
    pad_remover = expert_utils.PadRemover(targets_pad_mask)

  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepend_inputs_full_attention(
            targets_pad_mask))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias, pad_remover)


@expert_utils.add_name_scope()
def get_batch_coordinate(x, axis=0):
  """返回形状为 [1, batch_size*length, 1] 的扁平 int32 张量。

  参数：
      x: 输入张量
      axis: 坐标轴，默认为 0

  返回：
      批处理坐标张量
  """
  # Compute the batch coordinate before flattening all batches
  batch_coordinate = tf.expand_dims(
      common_attention.coordinate_tensor(tf.shape(x)[:-1], axis=axis), axis=-1)
  return batch_coordinate


@expert_utils.add_name_scope()
def expand_batch_coordinates(bc, length_factor):
  """将 bc 的元素按 length_factor 复制。

  参数：
      bc (tf.Tensor): 形状为 [1, length, 1] 的 int32 张量
      length_factor (int): 复制因子

  返回：
      tf.Tensor: 形状为 [1, length*length_factor, 1] 的张量，其中每个元素
          已被复制 length_factor 次
  """
  assert bc.get_shape().as_list() == [1, None, 1]
  # bc has shape [1, length, 1]
  bc *= tf.constant([[1] * length_factor])
  # bc has shape [1, length, length_factor]
  bc = tf.reshape(bc, [1, -1, 1])
  # bc has shape [1, length*length_factor]
  return bc


@expert_utils.add_name_scope()
def remove_pad(x, pad_remover, mode):
  """通过将所有维度连接成一个来移除填充。

  参数：
      x (tf.Tensor): 形状为 [batch_size, length, depth] 的输入
      pad_remover (obj): PadRemover 对象
      mode (ModeKeys): 推理、训练或评估。如果是推理，不应用填充移除器

  返回：
      形状为 [1,length_nonpad,depth] 的 tf.Tensor，其中
          length_nonpad <= batch_size*length
  """
  # Concatenate all tokens (without padding)
  x = expert_utils.flatten_all_but_last(x)

  # Remove padding for training and eval
  if mode != ModeKeys.PREDICT:
    # This is a hack to allows inference when the <go> token
    # is detected as padding and removed. This works for now because there is
    # no padding at inference.
    x = pad_remover.remove(x)

  x = tf.expand_dims(x, axis=0)  # Now batch_size=1
  return x


@expert_utils.add_name_scope()
def restore_pad(x, ref_x, pad_remover, mode):
  """恢复填充。

  参数：
      x: 输入张量
      ref_x: 参考张量，用于形状恢复
      pad_remover: PadRemover 对象
      mode: 模式

  返回：
      恢复填充后的张量
  """
  x = tf.squeeze(x, axis=0)
  if mode != ModeKeys.PREDICT:
    x = pad_remover.restore(x)
  x = common_layers.reshape_like(x, ref_x)
  return x


@registry.register_hparams
def attention_lm_moe_base():
  """基础超参数集。

  适用于 1 个 GPU。
  在 lm1b_32k 上：
     ~229M 参数
     在 [GeForce GTX TITAN X] 上 0.9 steps/sec

  返回：
      HParams 对象，包含注意力语言模型与 MoE 的基础超参数配置
  """
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 1024
  hparams.batch_size = 8192
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # 即无梯度裁剪
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 2000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 4
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = False
  hparams.add_hparam("filter_size", 2048)  # 像这样添加新的超参数
  hparams.moe_num_experts = 32
  # 注意力相关的标志
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # 所有以 "dropout" 结尾的超参数在非训练模式下会自动设置为 0.0
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("moe_layers", "2")  # 逗号分隔的层编号列表
  # moe 参数。局部注意力 moe。
  # 如果设置了 attention_layers，num_hidden_layers 参数将被忽略
  # 字符串的每个字符将对应一种注意力层类型
  hparams.add_hparam("attention_layers", "")
  hparams.add_hparam("attention_type", AttentionType.MULTIHEAD)
  hparams.add_hparam("attention_local", False)
  hparams.add_hparam("attention_moe_k", 2)
  hparams.add_hparam("attention_num_head", 1)
  hparams.add_hparam("attention_num_experts", 16)
  hparams.add_hparam("attention_split_batch", False)
  hparams.add_hparam("attention_red_factor", 3)
  hparams.add_hparam("attention_block_length", 128)
  hparams.add_hparam("attention_reduction_type", "conv")
  # 注意力减少的非线性。可以是 "none" 或 "silu"（
  # Sigmoid Linear-Unit，描述于 https://arxiv.org/abs/1710.05941）
  hparams.add_hparam("attention_nonlinearity", "none")
  # 如果设置了 attention_exp_factor，则每个输入到 local_expert_attention（维度为隐藏大小）
  # 会被投影到 attention_exp_factor 个更小的输入，每个维度为 attention_exp_inputdim。
  # （否则 attention_exp_inputdim 会被忽略）
  hparams.add_hparam("attention_exp_factor", 0)
  hparams.add_hparam("attention_exp_inputdim", 128)
  # 注意力的键、查询和值维度
  hparams.add_hparam("attention_kq_size", 128)
  hparams.add_hparam("attention_v_size", 256)
  # 负载平衡的损失系数
  hparams.add_hparam("attention_load_balance", 2e-2)
  # 局部敏感哈希参数
  hparams.add_hparam("lsh_num_hyperplanes", 4)
  hparams.add_hparam("lsh_use_map_fn", False)

  hparams.add_hparam("use_sepconv", False)
  hparams.add_hparam("diet_experts", False)
  hparams.add_hparam("memory_efficient_ffn", False)
  # if True, we learn a non-autoregressive model from "inputs" to "targets".
  # if False, we learn an autoregressive model to generate "targets"
  hparams.add_hparam("use_inputs", False)
  return hparams


@registry.register_hparams
def attention_lm_moe_base_long_seq():
  """长序列生成的特定超参数。

  返回：
      HParams 对象，包含长序列生成的超参数配置
  """
  hparams = attention_lm_moe_base()

  hparams.max_length = 0  # max_length == batch_size
  hparams.eval_drop_long_sequences = True
  hparams.min_length_bucket = 256  # 避免大批次的循环问题
  hparams.use_sepconv = True

  return hparams


@registry.register_hparams
def attention_lm_moe_base_ae():
  """带有注意力专家的基础模型。

  返回：
      HParams 对象，包含带有注意力专家的基础模型超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_type = AttentionType.LOCAL_EXPERTS

  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 10000
  # According to noam, ("n", "da") seems better for harder-to-learn models
  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_moe_base_local():
  """带有局部注意力的基础模型。

  返回：
      HParams 对象，包含带有局部注意力的基础模型超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_local = True
  return hparams


@registry.register_hparams
def attention_lm_moe_base_hybrid():
  """带有混合注意力的基础模型。

  返回：
      HParams 对象，包含带有混合注意力的基础模型超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "hehe"  # 交替使用局部/专家注意力
  hparams.attention_local = True

  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_hybrid_v2():
  """混合注意力模型的 v2 版本。

  返回：
      HParams 对象，包含混合注意力模型 v2 的超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "hheh"  # 交替使用局部/专家注意力
  hparams.attention_local = True
  hparams.attention_moe_k = 6

  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_16k():
  """批大小为 16k 的混合注意力模型。

  返回：
      HParams 对象，包含批大小为 16k 的混合注意力模型超参数配置
  """
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 16384
  return hparams


@registry.register_hparams
def attention_lm_12k():
  """批大小为 12k 的混合注意力模型。

  返回：
      HParams 对象，包含批大小为 12k 的混合注意力模型超参数配置
  """
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 12000
  return hparams


@registry.register_hparams
def attention_lm_11k():
  """批大小为 11k 的混合注意力模型。

  返回：
      HParams 对象，包含批大小为 11k 的混合注意力模型超参数配置
  """
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 11500
  return hparams


@registry.register_hparams
def attention_lm_ae_extended():
  """使用 exp_factor 参数的实验。

  返回：
      HParams 对象，包含使用 exp_factor 参数的实验模型超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "eeee"
  hparams.attention_local = True
  # hparams.factored_logits=1  # 当专家数量变大时必要
  hparams.attention_moe_k = 2
  hparams.attention_exp_factor = 4
  # hparams.attention_exp_inputdim = 128

  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_moe_base_memeff():
  """内存高效的基础模型。

  返回：
      HParams 对象，包含内存高效基础模型的超参数配置
  """
  hparams = attention_lm_moe_base_long_seq()
  hparams.use_sepconv = False

  hparams.diet_experts = True
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.memory_efficient_ffn = True
  hparams.attention_type = AttentionType.MEMORY_EFFICIENT
  hparams.num_heads = 8
  hparams.factored_logits = True
  return hparams


@registry.register_hparams
def attention_lm_moe_small():
  """单 GPU 训练的轻量级模型。

  在 lm1b_32k 上：
     ~312M 参数
     在 [GeForce GTX TITAN X] 上 1.6 steps/sec
     在 8 个 GPU 上同步训练 50K 步后：
        eval_log_ppl_per_token = 3.31

  返回：
      HParams 对象，包含轻量级注意力语言模型与 MoE 的超参数配置
  """
  hparams = attention_lm_moe_base()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.moe_num_experts = 128
  hparams.moe_layers = "2"
  return hparams


@registry.register_hparams
def attention_lm_moe_tiny():
  """用于调试的轻量级模型。

  返回：
      HParams 对象，包含用于调试的轻量级模型超参数配置
  """
  hparams = attention_lm_moe_small()
  hparams.moe_num_experts = 32
  return hparams


@registry.register_hparams
def attention_lm_attention_moe_tiny():
  """用于调试的注意力 MoE 轻量级模型。

  返回：
      HParams 对象，包含用于调试的注意力 MoE 轻量级模型超参数配置
  """
  hparams = attention_lm_moe_small()
  hparams.moe_layers = ""
  hparams.attention_num_experts = 128
  hparams.filter_size = 8192
  hparams.attention_type = AttentionType.LOCAL_EXPERTS
  return hparams


@registry.register_hparams
def attention_lm_no_moe_small():
  """无混合专家的模型（用于比较）。

  在 lm1b_32k 上：
     ~45M 参数
     在 [GeForce GTX TITAN X] 上 2 steps/sec
     在 8 个 GPU 上同步训练 50K 步后：
        eval_log_ppl_per_token = 3.51

  返回：
      HParams 对象，包含无混合专家的轻量级模型超参数配置
  """
  hparams = attention_lm_moe_small()
  hparams.moe_layers = ""
  return hparams


@registry.register_hparams
def attention_lm_moe_large():
  """用于分布式训练的大型模型。

  超过 1B 参数，因此由于内存要求需要多 GPU 训练。

  在 lm1b_32k 上：
     在 8 个 GPU 上同步训练 45K 步后：
        eval_log_ppl_per_token = 3.18
        eval_ppl_per_word = exp(1.107893 * eval_log_ppl_per_token) = 33.9

  返回：
      HParams 对象，包含大型注意力语言模型与 MoE 的超参数配置
  """
  hparams = attention_lm_moe_base()
  hparams.num_hidden_layers = 5
  hparams.moe_layers = "3"
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 4096
  hparams.moe_hidden_sizes = "4096"
  hparams.moe_num_experts = 128
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def attention_lm_moe_large_diet():
  """使用 diet experts 的大型模型。

  返回：
      HParams 对象，包含使用 diet experts 的大型模型超参数配置
  """
  hparams = attention_lm_moe_large()
  hparams.diet_experts = True
  return hparams


@registry.register_hparams
def attention_lm_moe_memory_efficient():
  """内存高效版本。

  返回：
      HParams 对象，包含内存高效版本的超参数配置
  """
  hparams = attention_lm_moe_large()
  hparams.diet_experts = True
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.memory_efficient_ffn = True
  hparams.attention_type = AttentionType.MEMORY_EFFICIENT
  hparams.num_heads = 8
  hparams.factored_logits = True
  return hparams


@registry.register_hparams
def attention_lm_moe_32b_diet():
  """具有 32B 参数的超大模型 - 因为我们可以。

  返回：
      HParams 对象，包含 32B 参数超大模型的超参数配置
  """
  hparams = attention_lm_moe_large_diet()
  hparams.moe_hidden_sizes = "16384"
  hparams.moe_num_experts = 1024
  return hparams


@registry.register_hparams
def attention_lm_moe_24b_diet():
  """具有 24B 参数的超大模型 - 因为我们可以。

  返回：
      HParams 对象，包含 24B 参数超大模型的超参数配置
  """
  hparams = attention_lm_moe_large_diet()
  hparams.moe_hidden_sizes = "12288"
  hparams.moe_num_experts = 1024
  hparams.batch_size = 4096
  return hparams


@registry.register_hparams
def attention_lm_moe_translation():
  """用于序列到序列任务的版本。

  返回：
      HParams 对象，包含用于序列到序列任务的注意力语言模型与 MoE 超参数配置
  """
  hparams = attention_lm_moe_base()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.learning_rate = 0.4
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 512
  hparams.label_smoothing = 0.1
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.num_hidden_layers = 6
  hparams.moe_layers = "0,1,2,3,4,5"
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams
def attention_lm_moe_unscramble_base():
  """用于 languagemodel_wiki_scramble1k50 的版本。

  返回：
      HParams 对象，包含用于文本解扰任务的注意力语言模型与 MoE 超参数配置
  """
  hparams = attention_lm_no_moe_small()
  hparams.use_inputs = True
  hparams.min_length_bucket = 1024
  hparams.max_length = 1024
  hparams.batch_size = 5000
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams
