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

"""简单的生成对抗模型，包含两个线性层。

T2T 中创建 GAN 的示例。

生成对抗网络（GAN）由生成器和判别器组成，
通过对抗训练来生成逼真的数据。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def deconv2d(
    input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
  """反卷积层。

  参数：
      input_: 输入张量
      output_shape: 输出形状
      k_h: 卷积核高度
      k_w: 卷积核宽度
      d_h: 高度步幅
      d_w: 宽度步幅
      stddev: 权重初始化的标准差
      name: 作用域名称

  返回：
      反卷积后的输出张量
  """
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(
        input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    biases = tf.get_variable(
        "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


def reverse_gradient(x):
  """反转梯度。

  用于梯度反转层，在对抗训练中常用。

  参数：
      x: 输入张量

  返回：
      梯度反转后的张量
  """
  return -x + tf.stop_gradient(2 * x)


class AbstractGAN(t2t_model.T2TModel):
  """所有 GAN 的基类。

  提供生成对抗网络的基础实现，包括判别器和生成器的基本架构。
  """

  def discriminator(self, x, is_training, reuse=False):
    """基于 InfoGAN 的判别器架构。

    参数：
        x: 输入图像，形状为 [bs, h, w, channels]
        is_training: 布尔值，表示是训练还是评估模式
        reuse: 布尔值，表示是否应重用参数

    返回：
        out_logit: 输出 logits（sigmoid 之前）
    """
    hparams = self.hparams
    with tf.variable_scope(
        "discriminator", reuse=reuse,
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size, height, width = common_layers.shape_list(x)[:3]
      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1")
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2")
      # [bs, h/4, w/4, 128]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn2")
      net = lrelu(net)
      size = height * width
      net = tf.reshape(net, [batch_size, size * 8])  # [bs, h * w * 8]
      net = tf.layers.dense(net, 1024, name="d_fc3")  # [bs, 1024]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn3")
      net = lrelu(net)
      return net

  def generator(self, z, is_training, out_shape):
    """Generator outputting image in [0, 1]."""
    hparams = self.hparams
    height, width, c_dim = out_shape
    batch_size = hparams.batch_size
    with tf.variable_scope(
        "generator",
        initializer=tf.random_normal_initializer(stddev=0.02)):
      net = tf.layers.dense(z, 1024, name="g_fc1")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn1")
      net = lrelu(net)
      net = tf.layers.dense(net, 128 * (height // 4) * (width // 4),
                            name="g_fc2")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [batch_size, height // 4, width // 4, 128])
      net = deconv2d(net, [batch_size, height // 2, width // 2, 64],
                     4, 4, 2, 2, name="g_dc3")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn3")
      net = lrelu(net)
      net = deconv2d(net, [batch_size, height, width, c_dim],
                     4, 4, 2, 2, name="g_dc4")
      out = tf.nn.sigmoid(net)
      return common_layers.convert_real_to_rgb(out)

  def losses(self, inputs, generated):
    """Return the losses dictionary."""
    raise NotImplementedError

  def body(self, features):
    """Body of the model.

    Args:
      features: a dictionary with the tensors.

    Returns:
      A pair (predictions, losses) where predictions is the generated image
      and losses is a dictionary of losses (that get added for the final loss).
    """
    features["targets"] = features["inputs"]
    is_training = self.hparams.mode == tf_estimator.ModeKeys.TRAIN

    # Input images.
    inputs = tf.to_float(features["targets_raw"])

    # Noise vector.
    z = tf.random_uniform([self.hparams.batch_size,
                           self.hparams.bottleneck_bits],
                          minval=-1, maxval=1, name="z")

    # Generator output: fake images.
    out_shape = common_layers.shape_list(inputs)[1:4]
    g = self.generator(z, is_training, out_shape)

    losses = self.losses(inputs, g)  # pylint: disable=not-callable

    summary_g_image = tf.reshape(
        g[0, :], [1] + common_layers.shape_list(inputs)[1:])
    tf.summary.image("generated", summary_g_image, max_outputs=1)

    if is_training:  # Returns an dummy output and the losses dictionary.
      return tf.zeros_like(inputs), losses
    return tf.reshape(g, tf.shape(inputs)), losses

  def top(self, body_output, features):
    """Override the top function to not do anything."""
    return body_output


@registry.register_model
class SlicedGan(AbstractGAN):
  """Sliced GAN for demonstration."""

  def losses(self, inputs, generated):
    """Losses in the sliced case."""
    is_training = self.hparams.mode == tf_estimator.ModeKeys.TRAIN
    def discriminate(x):
      return self.discriminator(x, is_training=is_training, reuse=False)
    generator_loss = common_layers.sliced_gan_loss(
        inputs, reverse_gradient(generated), discriminate,
        self.hparams.num_sliced_vecs)
    return {"training": - generator_loss}

  def infer(self, *args, **kwargs):  # pylint: disable=arguments-differ
    del args, kwargs

    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1

    with tf.variable_scope("body/vanilla_gan", reuse=tf.AUTO_REUSE):
      hparams = self.hparams
      z = tf.random_uniform([hparams.batch_size, hparams.bottleneck_bits],
                            minval=-1, maxval=1, name="z")
      out_shape = (hparams.sample_height, hparams.sample_width, num_channels)
      g_sample = self.generator(z, False, out_shape)
      return g_sample


@registry.register_hparams
def sliced_gan():
  """Basic parameters for a vanilla_gan."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 128
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.bottleneck_bits = 128
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 4096)
  return hparams
