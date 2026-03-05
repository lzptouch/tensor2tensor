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

"""从训练好的 T2T 模型进行解码。

该二进制文件使用 Estimator API 执行推理。

从数据集解码的示例用法：

  t2t-decoder \\
      --data_dir ~/data \\
      --problem=algorithmic_identity_binary40 \\
      --model=transformer
      --hparams_set=transformer_base

设置 FLAGS.decode_interactive 或 FLAGS.decode_from_file 以使用替代的解码源。

功能说明：
- 支持交互式本地推理
- 支持从文件批量解码
- 支持从数据集解码
- 支持对文件进行评分
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

flags = tf.flags
FLAGS = flags.FLAGS

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")
flags.DEFINE_bool("disable_grappler_optimizations", False,
                  "Disable Grappler if need be to avoid tensor format errors.")


def create_hparams():
  """创建超参数对象。
  
  Returns:
    HParams 对象，包含模型和数据集的配置
  
  功能说明：
  - 从输出目录加载超参数文件（如果存在）
  - 支持自定义超参数覆盖
  - 返回完整的超参数配置
  """
  hparams_path = None
  if FLAGS.output_dir:
    # 如果指定了输出目录，尝试从该目录加载已保存的超参数
    hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem,
      hparams_path=hparams_path)


def create_decode_hparams():
  """创建解码超参数对象。
  
  Returns:
    解码超参数对象，包含解码相关的配置参数
  
  功能说明：
  - 从 FLAGS 加载基础解码参数
  - 设置分片数量和分片 ID（用于分布式解码）
  - 配置输出文件和参考文件路径
  """
  # 从 FLAGS 加载基础解码参数
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  # 设置解码分片数量
  decode_hp.shards = FLAGS.decode_shards
  # 设置当前分片 ID
  decode_hp.shard_id = FLAGS.worker_id
  # 处理是否在内存中解码
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  # 设置输出文件路径
  decode_hp.decode_to_file = FLAGS.decode_to_file
  # 设置参考翻译文件路径（用于评估）
  decode_hp.decode_reference = FLAGS.decode_reference
  return decode_hp


def decode(estimator, hparams, decode_hp):
  """从 Estimator 进行解码。支持交互式、从文件或从数据集解码。
  
  Args:
    estimator: TensorFlow Estimator 对象
    hparams: 模型超参数
    decode_hp: 解码超参数
  
  功能说明：
  - 根据 FLAGS 选择不同的解码模式
  - 支持交互式本地推理（decode_interactive）
  - 支持从文件批量解码（decode_from_file）
  - 支持从标准数据集解码（默认模式）
  """
  if FLAGS.decode_interactive:
    # 交互式模式：用户输入文本，模型实时翻译
    if estimator.config.use_tpu:
      raise ValueError("TPU can only decode from dataset.")
    decoding.decode_interactively(estimator, hparams, decode_hp,
                                  checkpoint_path=FLAGS.checkpoint_path)
  elif FLAGS.decode_from_file:
    # 从文件解码：读取输入文件中的每一行进行翻译
    decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                              decode_hp, FLAGS.decode_to_file,
                              checkpoint_path=FLAGS.checkpoint_path)
    # 如果需要保持时间戳，则设置输出文件的时间戳与检查点一致
    if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
      ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
      os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
  else:
    # 从数据集解码：使用测试集或验证集进行批量解码
    decoding.decode_from_dataset(
        estimator,
        FLAGS.problem,
        hparams,
        decode_hp,
        decode_to_file=FLAGS.decode_to_file,
        dataset_split="test" if FLAGS.eval_use_test_set else None,
        checkpoint_path=FLAGS.checkpoint_path)


def score_file(filename):
  """Score each line in a file and return the scores."""
  # Prepare model.
  hparams = create_hparams()
  encoders = registry.problem(FLAGS.problem).feature_encoders(FLAGS.data_dir)
  has_inputs = "inputs" in encoders

  # Prepare features for feeding into the model.
  if has_inputs:
    inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
  targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
  batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
  if has_inputs:
    features = {"inputs": batch_inputs, "targets": batch_targets}
  else:
    features = {"targets": batch_targets}

  # Prepare the model and the graph when model runs on features.
  model = registry.model(FLAGS.model)(hparams, tf_estimator.ModeKeys.EVAL)
  _, losses = model(features)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load weights from checkpoint.
    if FLAGS.checkpoint_path is None:
      ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
      ckpt = ckpts.model_checkpoint_path
    else:
      ckpt = FLAGS.checkpoint_path
    saver.restore(sess, ckpt)
    # Run on each line.
    with tf.gfile.Open(filename) as f:
      lines = f.readlines()
    results = []
    for line in lines:
      tab_split = line.split("\t")
      if len(tab_split) > 2:
        raise ValueError("Each line must have at most one tab separator.")
      if len(tab_split) == 1:
        targets = tab_split[0].strip()
      else:
        targets = tab_split[1].strip()
        inputs = tab_split[0].strip()
      # Run encoders and append EOS symbol.
      targets_numpy = encoders["targets"].encode(
          targets) + [text_encoder.EOS_ID]
      if has_inputs:
        inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
      # Prepare the feed.
      if has_inputs:
        feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
      else:
        feed = {targets_ph: targets_numpy}
      # Get the score.
      np_loss = sess.run(losses["training"], feed)
      results.append(np_loss)
  return results


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)


  if FLAGS.score_file:
    filename = os.path.expanduser(FLAGS.score_file)
    if not tf.gfile.Exists(filename):
      raise ValueError("The file to score doesn't exist: %s" % filename)
    results = score_file(filename)
    if not FLAGS.decode_to_file:
      raise ValueError("To score a file, specify --decode_to_file for results.")
    write_file = tf.gfile.Open(os.path.expanduser(FLAGS.decode_to_file), "w")
    for score in results:
      write_file.write("%.6f\n" % score)
    write_file.close()
    return

  hp = create_hparams()
  decode_hp = create_decode_hparams()
  run_config = t2t_trainer.create_run_config(hp)
  if FLAGS.disable_grappler_optimizations:
    run_config.session_config.graph_options.rewrite_options.disable_meta_optimizer = True

  # summary-hook in tf.estimator.EstimatorSpec requires
  # hparams.model_dir to be set.
  hp.add_hparam("model_dir", run_config.model_dir)

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      run_config,
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  decode(estimator, hp, decode_hp)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
