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

"""使用给定目录中的所有检查点翻译文件。

将使用这些参数执行 t2t-decoder：
--problem
--data_dir
--output_dir 使用 --model_dir 的值
--decode_from_file 使用 --source 的值
--decode_hparams 使用正确格式的 --beam_size 和 --alpha
--checkpoint_path 自动填充
--decode_to_file 自动填充

功能说明：
- 批量使用多个检查点进行翻译
- 支持等待新检查点的持续翻译模式
- 自动生成翻译结果文件
- 集成 BLEU 评估功能
- 支持自定义解码器命令包装
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from tensor2tensor.utils import bleu_hook

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

# ========== t2t-translate-all 特定的配置选项 ==========
# 解码器命令模板（支持自定义包装器）
flags.DEFINE_string("decoder_command", "t2t-decoder {params}",
                    "要执行的命令，替代 t2t-decoder。"
                    "{params} 将被参数替换。例如用于 qsub 包装器。")
# 模型检查点目录
flags.DEFINE_string("model_dir", "",
                    "加载模型检查点的目录。")
# 源语言文件路径
flags.DEFINE_string("source", None,
                    "要翻译的源语言文件路径")
# 翻译结果存储目录
flags.DEFINE_string("translations_dir", "translations",
                    "存储翻译结果的目录。")
# 最小步数过滤
flags.DEFINE_integer("min_steps", 0, "忽略步数少于该值的检查点")
# 等待新检查点的时间（分钟）
flags.DEFINE_integer("wait_minutes", 0,
                     "等待新检查点的最长时间（分钟）")

# options derived from t2t-decoder
flags.DEFINE_integer("beam_size", 4, "Beam-search width.")
flags.DEFINE_float("alpha", 0.6, "Beam-search alpha.")
flags.DEFINE_string("model", "transformer", "see t2t-decoder")
flags.DEFINE_string("t2t_usr_dir", None, "see t2t-decoder")
flags.DEFINE_string("data_dir", None, "see t2t-decoder")
flags.DEFINE_string("problem", None, "see t2t-decoder")
flags.DEFINE_string("hparams_set", "transformer_big_single_gpu",
                    "see t2t-decoder")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # pylint: disable=unused-variable
  model_dir = os.path.expanduser(FLAGS.model_dir)
  translations_dir = os.path.expanduser(FLAGS.translations_dir)
  source = os.path.expanduser(FLAGS.source)
  tf.gfile.MakeDirs(translations_dir)
  translated_base_file = os.path.join(translations_dir, FLAGS.problem)

  # Copy flags.txt with the original time, so t2t-bleu can report correct
  # relative time.
  flags_path = os.path.join(translations_dir, FLAGS.problem + "-flags.txt")
  if not os.path.exists(flags_path):
    shutil.copy2(os.path.join(model_dir, "flags.txt"), flags_path)

  locals_and_flags = {"FLAGS": FLAGS}
  for model in bleu_hook.stepfiles_iterator(model_dir, FLAGS.wait_minutes,
                                            FLAGS.min_steps):
    tf.logging.info("Translating " + model.filename)
    out_file = translated_base_file + "-" + str(model.steps)
    locals_and_flags.update(locals())
    if os.path.exists(out_file):
      tf.logging.info(out_file + " already exists, so skipping it.")
    else:
      tf.logging.info("Translating " + out_file)
      params = (
          "--t2t_usr_dir={FLAGS.t2t_usr_dir} --output_dir={model_dir} "
          "--data_dir={FLAGS.data_dir} --problem={FLAGS.problem} "
          "--decode_hparams=beam_size={FLAGS.beam_size},alpha={FLAGS.alpha} "
          "--model={FLAGS.model} --hparams_set={FLAGS.hparams_set} "
          "--checkpoint_path={model.filename} --decode_from_file={source} "
          "--decode_to_file={out_file} --keep_timestamp"
      ).format(**locals_and_flags)
      command = FLAGS.decoder_command.format(**locals())
      tf.logging.info("Running:\n" + command)
      os.system(command)
  # pylint: enable=unused-variable


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
