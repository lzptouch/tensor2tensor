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

"""TransformerVaeFlowPrior 的各种操作。

实现 Transformer VAE 流先验模型中使用的各种操作，
包括编码器、解码器层、后验分布计算等。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_glow_layers_ops as gops
from tensor2tensor.models.transformer import transformer_decoder_layer
from tensor2tensor.models.transformer import transformer_encoder
from tensor2tensor.models.transformer import transformer_prepare_encoder
from tensor2tensor.utils import learning_rate as lr
from tensor2tensor.utils import mlperf_log
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


def _mixed_precision_is_enabled(hparams):
  """检查是否启用了混合精度。

  应该与 common_attention 中的相同，避免导入。

  参数：
      hparams: 超参数对象

  返回：
      布尔值，表示是否启用了混合精度
  """
  activation_dtype = hparams.activation_dtype
  weight_dtype = hparams.weight_dtype
  return activation_dtype == tf.float16 and weight_dtype == tf.float32


def encoder(name, hparams, inputs, target_space):
  """计算编码器输出和注意力偏置。

  参数：
      name: 变量作用域名称
      hparams: 超参数对象
      inputs: 输入张量
      target_space: 目标空间表示

  返回：
      encoder_output: 编码器输出
      encoder_decoder_attention_bias: 编码器 - 解码器注意力偏置
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    (encoder_input,
     encoder_self_attention_bias,
     encoder_decoder_attention_bias) = (
         transformer_prepare_encoder(inputs, target_space, hparams))
    encoder_input = tf.nn.dropout(encoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(encoder_input,
                                         encoder_self_attention_bias,
                                         hparams)
    return encoder_output, encoder_decoder_attention_bias


def transformer_decoder_layers(name,
                               n_layers,
                               decoder_input,
                               **kwargs):
  """由 transformer 解码器层组成的转换块。

  参数：
      name: 变量作用域名称
      n_layers: 解码器层数
      decoder_input: 解码器输入
      **kwargs: 其他参数，包含 hparams 等

  返回：
      outputs: 解码器输出
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    hparams = kwargs["hparams"]
    outputs = decoder_input
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for layer_idx in range(n_layers):
        outputs = transformer_decoder_layer(
            decoder_input=outputs,
            layer_idx=layer_idx,
            **kwargs)
      outputs = common_layers.layer_preprocess(outputs, hparams)
    return outputs


def posterior(
    name, hparams, targets, targets_mask, decoder_self_attention_bias,
    **kwargs):
  """计算对角正态后验分布 q(z|x,y) 的 mu 和 sigma。

  参数：
      name: 变量作用域名称
      hparams: 超参数对象
      targets: 目标序列
      targets_mask: 目标掩码
      decoder_self_attention_bias: 解码器自注意力偏置
      **kwargs: 其他参数

  返回：
      mu: 后验分布的均值
      sigma: 后验分布的标准差
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = drop_2d(targets, hparams.mode, hparams.posterior_2d_dropout)
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    decoder_input = tf.nn.dropout(decoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_posterior_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    decoder_output = gops.dense_weightnorm(
        "h2o_out", decoder_output, hparams.latent_size * 2, targets_mask,
        init_scale=0.0, init=False)
    return decoder_output


def cond_prior(
    name, hparams, decoder_input, targets_mask, output_size,
    decoder_self_attention_bias, init_scale=0.0, **kwargs):
  """计算条件先验的参数隐藏状态。

  参数：
      name: 变量作用域名称
      hparams: 超参数对象
      decoder_input: 解码器输入
      targets_mask: 目标掩码
      output_size: 输出维度
      decoder_self_attention_bias: 解码器自注意力偏置
      init_scale: 初始化缩放因子
      **kwargs: 其他参数

  返回：
      decoder_output: 条件先验的隐藏状态
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    decoder_input = tf.nn.dropout(decoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_posterior_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    decoder_output = gops.dense_weightnorm(
        "h2o_out", decoder_output, output_size, targets_mask,
        init_scale=init_scale, init=False)
    return decoder_output


def decoder(name, latents, hparams, decoder_self_attention_bias, **kwargs):
  """计算 p(y|z,x) 的最终隐藏状态。

  参数：
      name: 变量作用域名称
      latents: 潜在变量
      hparams: 超参数对象
      decoder_self_attention_bias: 解码器自注意力偏置
      **kwargs: 其他参数

  返回：
      decoder_output: 解码器最终隐藏状态
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = drop_2d(latents, hparams.mode, hparams.decoder_2d_dropout)
    if hparams.pos_attn:
      decoder_input = gops.positional_attention(
          "pos_attn", decoder_input, decoder_self_attention_bias, hparams)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    if common_layers.shape_list(latents)[-1] != hparams.hidden_size:
      decoder_input = gops.dense("lat2hid", latents, hparams.hidden_size)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_decoder_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    batch_size, targets_length = common_layers.shape_list(decoder_output)[:2]
    decoder_output = tf.reshape(
        decoder_output, [batch_size, targets_length, 1, hparams.hidden_size])
    # Expand since t2t expects 4d tensors.
    return decoder_output


def drop_2d(targets, mode, dropout_p):
  """2D Dropout 操作。

  参数：
      targets: 目标张量
      mode: 模型模式（TRAIN/EVAL/PREDICT）
      dropout_p: dropout 概率

  返回：
      targets_noisy: 应用 dropout 后的张量
  """
  if dropout_p > 0 and mode == tf_estimator.ModeKeys.TRAIN:
    batch_size, targets_length, hidden_size = common_layers.shape_list(targets)
    mask_prob = tf.random_uniform(
        shape=(batch_size, targets_length), minval=0.0, maxval=1.0)
    mask_prob = tf.tile(mask_prob[..., tf.newaxis], [1, 1, hidden_size])
    scale = 1 / (1 - dropout_p)
    targets_noisy = tf.where(
        mask_prob > dropout_p, targets * scale, tf.zeros_like(targets))
    return targets_noisy
  return targets


def sequence_mask(length, hparams):
  """生成序列掩码。

  参数：
      length: 序列长度
      hparams: 超参数对象

  返回：
      序列掩码张量
  """
  dtype = get_dtype(hparams)
  return tf.sequence_mask(length, dtype=dtype)


def get_padding(mask, hparams):
  """获取填充掩码。

  参数：
      mask: 输入掩码
      hparams: 超参数对象

  返回：
      填充掩码张量
  """


def get_padding(mask, hparams):
  """获取填充掩码。

  参数：
      mask: 输入掩码
      hparams: 超参数对象

  返回：
      填充掩码张量
  """
  dtype = get_dtype(hparams)
  return tf.cast(tf.equal(mask, 0.0), dtype=dtype)


def get_dtype(hparams):
  """获取数据类型。

  参数：
      hparams: 超参数对象

  返回：
      TensorFlow 数据类型
  """
  if hparams.activation_dtype == "float32":
    return tf.float32
  elif hparams.activation_dtype == "float64":
    return tf.float64
  elif hparams.activation_dtype == "bfloat16":
    return tf.bfloat16
  else:
    return None


def lenpred_mlp(name, logits, hidden_size, bound):
  """长度预测多层感知机。

  参数：
      name: 变量作用域名称
      logits: 输入 logits
      hidden_size: 隐藏层大小
      bound: 边界值

  返回：
      logits: 预测的 logits
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    logits = tf.layers.dense(logits, hidden_size)
    logits = tf.nn.elu(logits)
    logits = tf.layers.dense(logits, hidden_size)
    logits = tf.nn.elu(logits)
    logits = tf.layers.dense(logits, bound * 2 + 1)
  return logits


def predict_target_lengths(
    encoder_output, inputs_mask, hparams, length_diff=None):
  """预测目标序列长度。

  参数：
      encoder_output: 编码器输出
      inputs_mask: 输入掩码
      hparams: 超参数对象
      length_diff: 长度差异（可选）

  返回：
      预测的目标长度
  """
  bound = hparams.lendiff_bound
  inputs_length = tf.cast(tf.reduce_sum(inputs_mask, 1), tf.int32)
  targets_length = inputs_length
  loss = None
  if hparams.predict_target_length:
    encoder_output = gops.reduce_mean_over_l(encoder_output, inputs_mask)
    logits = tf.stop_gradient(encoder_output)
    logits = lenpred_mlp("lenpred", logits, hparams.hidden_size, bound)
    if length_diff is not None:
      labels = tf.maximum(tf.minimum(length_diff, bound), -bound)
      labels = tf.cast(labels + bound, tf.int32)
      labels = tf.stop_gradient(labels)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(loss)
    diff_pred = tf.argmax(logits, 1)
    diff_pred = tf.cast(diff_pred - bound, tf.int32)
    targets_length = inputs_length + diff_pred
    targets_length = tf.maximum(targets_length, 1)
  divi = 4
  targets_length = tf.ceil(targets_length / divi) * divi
  targets_length = tf.cast(targets_length, tf.int32)
  return targets_length, loss


def lenpred_stats(targets_length_pred, targets_length):
  """计算长度预测的统计信息。

  参数：
      targets_length_pred: 预测的目标长度
      targets_length: 真实目标长度

  返回：
      lenpred_acc: 精确匹配准确率
      lenpred_acc5: 误差在 5 以内的准确率
  """
  lenpred_diff = tf.abs(targets_length_pred - tf.cast(targets_length, tf.int32))
  lenpred_acc = tf.cast(tf.equal(lenpred_diff, 0), tf.float32)
  lenpred_acc = tf.reduce_mean(lenpred_acc)
  lenpred_acc5 = tf.cast(tf.less_equal(lenpred_diff, 5), tf.float32)
  lenpred_acc5 = tf.reduce_mean(lenpred_acc5)
  return lenpred_acc, lenpred_acc5


def save_log_loss(
    hparams, targets_mask, numerator, denominator, log_q_z, log_abs_det,
    log_p_z_base, z_q, lenpred_loss, targets_length_pred, targets_length):
  """填充损失字典和摘要。

  参数：
      hparams: 超参数对象
      targets_mask: 目标掩码
      numerator: 分子
      denominator: 分母
      log_q_z: 后验分布的对数概率
      log_abs_det: 雅可比行列式的对数绝对值
      log_p_z_base: 基础先验的对数概率
      z_q: 后验分布的潜在变量
      lenpred_loss: 长度预测损失
      targets_length_pred: 预测的目标长度
      targets_length: 真实目标长度

  返回：
      loss_dict: 损失字典
      monitor: 监控指标字典
  """
  anneal, kl_mask = get_anneal_mask(hparams)
  lenpred_acc, lenpred_acc5 = (
      lenpred_stats(targets_length_pred, targets_length))
  batch_length = tf.reduce_sum(targets_mask)

  z_q_norm = gops.reduce_mean_over_bl(
      tf.norm(z_q, axis=2, keepdims=True), targets_mask)[0]

  log_q_z = gops.reduce_mean_over_bl_sum_over_c(log_q_z, targets_mask)
  log_p_z_base = tf.reduce_sum(log_p_z_base, axis=0) / batch_length
  log_abs_det = tf.reduce_sum(log_abs_det, axis=0) / batch_length
  log_p_z_reg = gops.standard_normal_density(z_q, targets_mask, reduce_sum=True)

  log_p_x = -1 * numerator / denominator
  log_p_z = log_p_z_base + log_abs_det
  kl = log_q_z - log_p_z
  kl_reg = log_p_z - log_p_z_reg
  elbo = log_p_x - kl
  monitor = {
      "elbo": elbo,
      "kl": kl,
      "kl_reg": kl_reg,
      "log_p_x": log_p_x,
      "log_q_z": log_q_z,
      "log_p_z": log_p_z,
      "log_p_z_base": log_p_z_base,
      "log_abs_det": log_abs_det,
      "anneal": anneal,
      "z_q_norm": z_q_norm,
      "lenpred_acc": lenpred_acc,
      "lenpred_acc5": lenpred_acc5,
  }

  kl = kl * anneal
  kl_reg = hparams.kl_reg * kl_reg * anneal
  loss_dict = {
      "training": -1 * log_p_x,
      "kl": kl * kl_mask,
      "kl_reg": kl_reg * kl_mask,
  }
  if lenpred_loss is not None:
    monitor["lenpred_loss"] = lenpred_loss
    loss_dict["lenpred_loss"] = lenpred_loss
  return loss_dict, monitor


def get_anneal_mask(hparams):
  """获取退火掩码和 KL 掩码。

  参数：
      hparams: 超参数对象，包含 kl_startup_steps 和 kl_anneal_steps

  返回：
      anneal: 退火因子
      kl_mask: KL 散度掩码
  """
  startup = hparams.kl_startup_steps
  anneal = hparams.kl_anneal_steps
  global_step = tf.train.get_global_step()
  min_value = hparams.anneal_min_value
  step = tf.maximum(global_step - startup, 0)
  anneal = common_layers.inverse_lin_decay(
      anneal, min_value=min_value, step=step)
  kl_mask = tf.less(startup, tf.to_int32(global_step))
  kl_mask = tf.cast(kl_mask, tf.float32)
  return anneal, kl_mask


def embedding_to_non_padding(emb, dtype=tf.float32):
  """根据嵌入是否为零计算填充掩码。

  参数：
      emb: 嵌入张量
      dtype: 数据类型，默认为 tf.float32

  返回：
      填充掩码张量
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.cast(tf.not_equal(emb_sum, 0.0), dtype=dtype)


def save_summary(monitor, name):
  """保存摘要。

  参数：
      monitor: 监控指标字典
      name: 摘要名称
  """
  with tf.name_scope(name):
    for key in list(monitor.keys()):
      tf.summary.scalar(key, monitor[key])


def _global_step(hparams):
  """如果使用了多步优化器，则调整全局步数。

  参数：
      hparams: 超参数对象

  返回：
      调整后的全局步数
  """
  step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
  multiplier = hparams.optimizer_multistep_accumulate_steps
  if not multiplier:
    return step

  tf.logging.info("Dividing global step by %d for multi-step optimizer."
                  % multiplier)
  return step / tf.cast(multiplier, tf.float32)


def learning_rate_schedule(hparams):
  """基于超参数的学习率调度。

  参数：
      hparams: 超参数对象

  返回：
      学习率张量
  """
  mlperf_log.transformer_print(key=mlperf_log.OPT_LR, deferred=True)
  mlperf_log.transformer_print(
      key=mlperf_log.OPT_LR_WARMUP_STEPS,
      value=hparams.learning_rate_warmup_steps)
  step_num = _global_step(hparams)
  # Simulate pretraining the encoder, decoder and posterior with the same
  # learning rate schedule, and then restoring the parameters.
  # using `warm_start_from` is not compatible with actnorm DDI on TPUs.
  step_num = tf.where(
      step_num < hparams.kl_startup_steps,
      step_num,
      step_num - hparams.kl_startup_steps)
  schedule_string = hparams.learning_rate_schedule
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= lr.learning_rate_factor(name, step_num, hparams)
  return ret


def prepare_for_iw(x, k):
  """为重要性采样准备特征。

  参数：
      x: 输入张量
      k: 采样次数

  返回：
      扩展后的张量，形状为 [k, batch_size, ...]
  """
  batch_size = common_layers.shape_list(x)[0]
  remaining_shape = common_layers.shape_list(x)[1:]

  multiplier = [1] * x.shape.rank
  x = tf.tile(x[tf.newaxis, ...], [k] + multiplier)
  x = tf.reshape(x, [k * batch_size] + remaining_shape)
  return x


def unprepare_for_iw(x, k):
  """为重要性采样还原特征。

  参数：
      x: 输入张量，形状为 [k * batch_size, ...]
      k: 采样次数

  返回：
      还原后的张量，形状为 [k, batch_size, ...]
  """
  batch_size_times_k = common_layers.shape_list(x)[0]
  remaining_shape = common_layers.shape_list(x)[1:]
  x = tf.reshape(x, [k, batch_size_times_k // k] + remaining_shape)
  return x


def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """计算一个输出分片的损失分子和分母。

  参数：
      top_out: 模型输出
      targets: 目标序列
      model_hparams: 模型超参数
      vocab_size: 词汇表大小（未使用）
      weights_fn: 权重函数

  返回：
      损失分子和分母
  """
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn,
      reduce_sum=False)
