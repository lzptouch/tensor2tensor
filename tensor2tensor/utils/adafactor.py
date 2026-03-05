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

"""优化算法。

实现 Adafactor 优化器及其相关功能。

功能说明：
- 实现 Adafactor 优化算法
- 支持内存高效的二阶矩估计
- 提供更新裁剪机制增强稳定性
- 适用于大规模深度学习模型训练
- 支持量化感知训练
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import quantization

import tensorflow.compat.v1 as tf


class AdafactorOptimizer(tf.train.Optimizer):
  """实现 Adafactor 算法的优化器。
  
  Adafactor 在 https://arxiv.org/abs/1804.04235 中描述。
  
  Adafactor 与 Adam（Kingma 和 Ba）最相似，主要区别在于：
  
  1. 对于 AxB 的二维权重矩阵，Adafactor 只使用 A+B 个辅助参数来维护
     二阶矩估计器，而不是 AB 个。这在内存受限的系统上很有优势。
     此外，默认情况下 beta1（动量）设置为零，节省了每个权重的额外辅助参数。
     具有 >=3 维的变量被视为二维矩阵的集合——因式分解在最后两个维度上进行。
  
  2. Adafactor 结合了"更新裁剪"——梯度裁剪的尺度不变类比。这增加了稳定性
  
  3. Adafactor 不需要外部的"学习率"。默认情况下，它结合了相对更新尺度计划，
     对应于 ADAM 中的逆平方根学习率衰减。我们希望这对大多数应用都有效。
  
  算法：
  
      parameter -= absolute_update_scale * clip(grad / grad_scale)
  
  其中：
  
      absolute_update_scale := relative_update_scale * parameter_scale
      relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
      parameter_scale := max(rms(var), epsilon2)
      clip(x) := x / max(1.0, rms(x))
      grad_scale := tf.sqrt(v)   (v 是二阶矩估计器)
  
    二阶矩估计器 v 的维护方式与 Adam 类似：
    我们初始化
    ```
    如果 var 是二维的：
      v_r <- zeros([num_rows])
      v_c <- zeros([num_cols])
    如果 var 是 0 维或 1 维的：
      v <- zeros(shape(var))
    ```
  
    更新规则如下：
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon1
    如果 var 是二维的：
      v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
      v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
      v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    如果 var 是 0 维或 1 维的：
      v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
  
    对于具有 >=3 维的变量，我们在最后 2 个维度上对二阶矩累加器进行因式分解。
    有关详细信息，请参阅代码。
  
  
    该算法的几个部分可以从初始化器配置。
  
  功能说明：
  - 内存高效的优化器（相比 Adam 节省内存）
  - 自动调整学习率
  - 支持梯度裁剪和更新裁剪
  - 适用于 Transformer 等大型模型训练
  
      multiply_by_parameter_scale: 如果为 True，则按上述方式计算 absolute_update_scale。
        如果为 False，则让 absolute_update_scale 为外部提供的 learning_rate。
      learning_rate: 如果 multiply_by_parameter_scale==True，则表示 relative_update_scale，
        如果 multiply_by_parameter_scale==False，则表示 absolute_update_scale。
      decay_rate: 二阶矩估计器的衰减率（随 step_num 变化）。
        这应该设置为一个函数，使得：
        1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
      beta1: 启用动量，如 Adam。如果非零则使用额外内存。
      clipping_threshold: 应该 >=1.0 或 None 表示不进行更新裁剪
      factored: 是否对二阶矩估计器进行因式分解。True 表示更少的内存使用。
  """

  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               simulated_quantize_bits=None,
               parameter_encoding=None,
               use_locking=False,
               name="Adafactor",
               epsilon1=1e-30,
               epsilon2=1e-3):
    """构造一个新的 Adafactor 优化器。

    参见类注释。

    参数：
        multiply_by_parameter_scale: 布尔值
        learning_rate: 可选的标量或可调用对象
        decay_rate: 可选的标量
        beta1: 0 到 1 之间的浮点值
        clipping_threshold: 可选的浮点数 >= 1
        factored: 布尔值 - 是否对 2d 变量使用因式分解的二阶矩估计器
        simulated_quantize_bits: 使用模拟量化参数进行训练（实验性）
        parameter_encoding: 在 bfloat16 变量情况下使用的 ParameterEncoding 对象
        use_locking: 如果为 True，则在更新操作时使用锁
        name: 创建应用梯度的操作时的可选名称。默认为"AdafactorOptimizer"
        epsilon1: 平方梯度的正则化常数
        epsilon2: 参数尺度的正则化常数

    异常：
        ValueError: 如果 absolute_update_scale 和 relative_update_scale_fn 都存在或都不存在
    """
    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._simulated_quantize_bits = simulated_quantize_bits
    self._parameter_encoding = parameter_encoding
    self._quantization_noise = quantization.noise_from_step_num()
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2

  def _should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers
    Returns:
      a boolean
    """
    return self._factored and len(shape) >= 2

  def _create_slots(self, var_list):
    for var in var_list:
      shape = var.get_shape().as_list()
      if self._beta1:
        self._zeros_slot(var, "m", self._name)
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros(shape[:-1], dtype=tf.float32)
        c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
        self._get_or_make_slot(var, r_val, "vr", self._name)
        self._get_or_make_slot(var, c_val, "vc", self._name)
      else:
        v_val = tf.zeros(shape, dtype=tf.float32)
        self._get_or_make_slot(var, v_val, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    return self._apply_dense(tf.convert_to_tensor(grad), var)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.
    Returns:
      a Scalar
    """
    return tf.maximum(reduce_rms(var), self._epsilon2)

  def _resource_apply_dense(self, grad, handle):
    var = handle
    grad = tf.to_float(grad)
    grad_squared = tf.square(grad) + self._epsilon1
    grad_squared_mean = tf.reduce_mean(grad_squared)
    decay_rate = self._call_if_callable(self._decay_rate)
    update_scale = self._call_if_callable(self._learning_rate)
    update_scale = tf.convert_to_tensor(update_scale, name="update_scale")
    update_scale = tf.cast(update_scale, grad_squared_mean.dtype.base_dtype)
    old_val = var
    if var.dtype.base_dtype == tf.bfloat16:
      old_val = tf.to_float(self._parameter_encoding.decode(old_val))
    if self._multiply_by_parameter_scale:
      update_scale *= tf.to_float(self._parameter_scale(old_val))
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = 1.0 - decay_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
      r_factor = tf.rsqrt(new_vr / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      x = grad * tf.rsqrt(new_v)
    if self._clipping_threshold is not None:
      clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      m = self.get_slot(var, "m")
      new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
      subtrahend = new_m
      new_m = common_layers.cast_like(new_m, var)
      updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
    new_val = tf.to_float(old_val) - subtrahend
    if var.dtype.base_dtype == tf.bfloat16:
      new_val = self._parameter_encoding.encode(
          new_val, self._quantization_noise)
    if self._simulated_quantize_bits:
      new_val = quantization.simulated_quantize(
          var - subtrahend, self._simulated_quantize_bits,
          self._quantization_noise)
    new_val = tf.cast(new_val, var.dtype)
    var_update = tf.assign(var, new_val, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
    if not multiply_by_parameter_scale:
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: a float between 0 and 1
  Returns:
    a scalar
  """
  t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
  return decay


def adafactor_decay_rate_pow(exponent):
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: a float between 0 and 1
  Returns:
    a scalar
  """
  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
  """Create an Adafactor optimizer based on model hparams.

  Args:
    hparams: model hyperparameters
    lr: learning rate scalar.
  Returns:
    an AdafactorOptimizer
  Raises:
    ValueError: on illegal values
  """
  if hparams.optimizer_adafactor_decay_type == "adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  if hparams.weight_dtype == "bfloat16":
    parameter_encoding = quantization.EighthPowerEncoding()
  else:
    parameter_encoding = None
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored,
      simulated_quantize_bits=getattr(
          hparams, "simulated_parameter_quantize_bits", 0),
      parameter_encoding=parameter_encoding,
      use_locking=False,
      name="Adafactor")


def reduce_rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))
