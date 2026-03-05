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

"""围绕 tf.contrib 的包装器，用于动态导入 contrib 包。

这确保依赖 T2T 和 TF2 的库在导入时不会崩溃。

功能说明：
- 提供 TensorFlow 1.x 和 2.x 的兼容性层
- 动态导入 tf.contrib.slim
- 在 TF2 中使用 tensorflow_addons 替代
- 确保代码在不同 TF 版本间可移植
"""

from __future__ import absolute_import
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

# 检查是否有 contrib 可用
try:
  # TensorFlow 1.x：直接导入 contrib.slim
  from tensorflow.contrib import slim as tf_slim  # pylint: disable=g-import-not-at-top
  is_tf2 = False
except:  # pylint: disable=bare-except
  # TensorFlow 2.x：tf.contrib.slim 不可用，使用替代包
  # 一些功能现在在单独的包中提供。我们根据需要支持这些。
  import tensorflow_addons as tfa  # pylint: disable=g-import-not-at-top
  import tf_slim  # pylint: disable=g-import-not-at-top
  is_tf2 = True


def err_if_tf2(msg='err'):
  if is_tf2:
    if msg == 'err':
      msg = 'contrib is unavailable in tf2.'
      raise ImportError(msg)
    else:
      msg = 'contrib is unavailable in tf2.'
      logging.info(msg)


class DummyModule(object):

  def __init__(self, **kw):
    for k, v in kw.items():
      setattr(self, k, v)


def slim():
  return tf_slim


def util():
  err_if_tf2()
  from tensorflow.contrib import util as contrib_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_util


def tfe():
  err_if_tf2(msg='warn')
  from tensorflow.contrib.eager.python import tfe as contrib_eager  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_eager


def deprecated(reason, date):
  del reason
  del date
  def decorator(fn):
    return fn
  return decorator


def framework(msg='err'):
  """Return framework module or dummy version."""
  del msg
  if is_tf2:
    return DummyModule(
        arg_scope=None,
        get_name_scope=lambda: tf.get_default_graph().get_name_scope(),
        name_scope=tf.name_scope,
        deprecated=deprecated,
        nest=tf.nest,
        argsort=tf.argsort)

  from tensorflow.contrib import framework as contrib_framework  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_framework


def nn():
  err_if_tf2(msg='err')
  from tensorflow.contrib import nn as contrib_nn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_nn


def layers():
  """Return layers module or dummy version."""
  try:
    from tensorflow.contrib import layers as contrib_layers  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    return contrib_layers
  except:  # pylint: disable=bare-except
    return DummyModule(
        OPTIMIZER_CLS_NAMES={}, optimize_loss=tf_slim.optimize_loss)


def rnn():
  err_if_tf2(msg='err')
  from tensorflow.contrib import rnn as contrib_rnn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_rnn


def seq2seq():
  err_if_tf2(msg='err')
  from tensorflow.contrib import seq2seq as contrib_seq2seq  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_seq2seq


def tpu():
  err_if_tf2(msg='err')
  from tensorflow.contrib import tpu as contrib_tpu  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_tpu


def training():
  err_if_tf2(msg='err')
  from tensorflow.contrib import training as contrib_training  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_training


def summary():
  err_if_tf2(msg='err')
  from tensorflow.contrib import summary as contrib_summary  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_summary


def metrics():
  err_if_tf2(msg='err')
  from tensorflow.contrib import metrics as contrib_metrics  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_metrics


def opt():
  if not is_tf2:
    from tensorflow.contrib import opt as contrib_opt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    return contrib_opt
  return DummyModule(
      LazyAdam=tfa.optimizers.LazyAdam,
      LazyAdamOptimizer=tfa.optimizers.LazyAdam,
  )


def mixed_precision():
  err_if_tf2(msg='err')
  from tensorflow.contrib import mixed_precision as contrib_mixed_precision  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_mixed_precision


def cluster_resolver():
  err_if_tf2(msg='err')
  from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_cluster_resolver


def distribute():
  err_if_tf2(msg='err')
  from tensorflow.contrib import distribute as contrib_distribute  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_distribute


def replace_monitors_with_hooks(monitors_or_hooks, estimator):
  """Stub for missing function."""
  del estimator
  monitors_or_hooks = monitors_or_hooks or []
  hooks = [
      m for m in monitors_or_hooks if isinstance(m, tf_estimator.SessionRunHook)
  ]
  deprecated_monitors = [
      m for m in monitors_or_hooks
      if not isinstance(m, tf_estimator.SessionRunHook)
  ]
  assert not deprecated_monitors
  return hooks


def learn():
  """Return tf.contrib.learn module or dummy version."""
  if not is_tf2:
    from tensorflow.contrib import learn as contrib_learn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    return contrib_learn
  return DummyModule(
      RunConfig=tf_estimator.RunConfig,
      monitors=DummyModule(
          replace_monitors_with_hooks=replace_monitors_with_hooks),
  )


def tf_prof():
  err_if_tf2(msg='err')
  from tensorflow.contrib import tfprof as contrib_tfprof  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_tfprof


def eager():
  err_if_tf2(msg='err')
  from tensorflow.contrib.eager.python import tfe as contrib_eager  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_eager


def image():
  err_if_tf2(msg='err')
  from tensorflow.contrib import image as contrib_image  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_image
