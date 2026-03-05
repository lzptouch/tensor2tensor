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

"""评估给定目录中所有检查点/翻译的 BLEU 分数。

此脚本有两种使用方式。

评估一个已经翻译的文件：

```
t2t-bleu --translation=my-wmt13.de --reference=wmt13_deen.de
```

评估给定目录中的所有翻译（由 `t2t-translate-all` 翻译）：

```
t2t-bleu
  --translations_dir=my-translations
  --reference=wmt13_deen.de
  --event_dir=events
```

除了上述必需参数外，还有可选参数：
 * bleu_variant: cased（区分大小写）、uncased、both（默认）
 * tag_suffix: 默认=""，因此标签将为 BLEU_cased 和 BLEU_uncased。
   tag_suffix 可用于不同的 beam size，如果这些应该在不同的图中绘制。
 * min_steps: 不评估步数较少的检查点。
   默认=-1 表示检查 `last_evaluated_step.txt` 文件，其中包含最后一次成功评估的检查点的步数。
 * report_zero: 存储 BLEU=0 并根据 translations_dir 中最旧的文件猜测其时间。默认=True。
   这很有用，因此 TensorBoard 可以为剩余检查点报告正确的相对时间。
   如果 min_steps > 0，则此标志设置为 False。
 * wait_minutes: 等待新翻译文件最多 N 分钟。默认=0。
   这对于运行训练的连续评估很有用，在这种情况下，
   这应该等于 save_checkpoints_secs/60 加上翻译所需时间加上一些余量。

功能说明：
- 计算翻译结果与参考翻译之间的 BLEU 分数
- 支持区分大小写和不区分大小写两种模式
- 支持单个文件或批量文件的评估
- 将评估结果写入 TensorBoard 事件文件
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from tensor2tensor.utils import bleu_hook
import tensorflow.compat.v1 as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("source", None,
                    "Path to the source-language file to be translated")
flags.DEFINE_string("reference", None, "Path to the reference translation file")
flags.DEFINE_string("translation", None,
                    "Path to the MT system translation file")
flags.DEFINE_string("translations_dir", None,
                    "Directory with translated files to be evaluated.")
flags.DEFINE_string("event_dir", None, "Where to store the event file.")

flags.DEFINE_string("bleu_variant", "both",
                    "Possible values: cased(case-sensitive), uncased, "
                    "both(default).")
flags.DEFINE_string("tag_suffix", "",
                    "What to add to BLEU_cased and BLEU_uncased tags.")
flags.DEFINE_integer("min_steps", -1,
                     "Don't evaluate checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint, cf. "
                     "save_checkpoints_secs.")
flags.DEFINE_bool("report_zero", None,
                  "Store BLEU=0 and guess its time based on the oldest file.")


def main(_):
  """主函数：执行 BLEU 分数评估。
  
  Args:
    _: 未使用的参数占位符
  
  流程说明：
  1. 检查是单文件模式还是批量模式
  2. 单文件模式：直接计算并输出 BLEU 分数
  3. 批量模式：遍历所有翻译文件，逐个评估并写入 TensorBoard
  """
  # 设置日志级别
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # 单文件模式：评估单个翻译文件
  if FLAGS.translation:
    # 不能同时指定 translation 和 translations_dir
    if FLAGS.translations_dir:
      raise ValueError(
          "Cannot specify both --translation and --translations_dir.")
    
    # 根据 bleu_variant 选择评估模式
    if FLAGS.bleu_variant in ("uncased", "both"):
      # 不区分大小写的 BLEU 评估
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
                                          case_sensitive=False)
      print("BLEU_uncased = %6.2f" % bleu)
    if FLAGS.bleu_variant in ("cased", "both"):
      # 区分大小写的 BLEU 评估
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
                                          case_sensitive=True)
      print("BLEU_cased = %6.2f" % bleu)
    return

  # 批量模式：评估翻译目录中的所有文件
  if not FLAGS.translations_dir:
    raise ValueError(
        "Either --translation or --translations_dir must be specified.")
  
  # 展开翻译目录路径
  transl_dir = os.path.expanduser(FLAGS.translations_dir)
  
  # 如果翻译目录不存在，等待指定时间
  if not os.path.exists(transl_dir):
    exit_time = time.time() + FLAGS.wait_minutes * 60
    tf.logging.info("Translation dir %s does not exist, waiting till %s.",
                    transl_dir, time.asctime(time.localtime(exit_time)))
    while not os.path.exists(transl_dir):
      time.sleep(10)
      if time.time() > exit_time:
        raise ValueError("Translation dir %s does not exist" % transl_dir)

  # 获取上次评估的步数
  last_step_file = os.path.join(FLAGS.event_dir, "last_evaluated_step.txt")
  if FLAGS.min_steps == -1:
    # 如果未指定 min_steps，从文件中读取上次成功评估的步数
    if tf.gfile.Exists(last_step_file):
      with open(last_step_file) as ls_file:
        FLAGS.min_steps = int(ls_file.read())
    else:
      FLAGS.min_steps = 0
  
  # 确定是否报告 BLEU=0（用于 TensorBoard 时间轴）
  if FLAGS.report_zero is None:
    FLAGS.report_zero = FLAGS.min_steps == 0

  # 创建 TensorBoard 事件写入器
  writer = tf.summary.FileWriter(FLAGS.event_dir)
  
  # 遍历所有翻译文件（按步数排序）
  for transl_file in bleu_hook.stepfiles_iterator(
      transl_dir, FLAGS.wait_minutes, FLAGS.min_steps, path_suffix=""):
    
    # report_zero 处理：在第一个点之前添加 BLEU=0 的点
    if FLAGS.report_zero:
      # 获取翻译目录中所有文件的最早时间戳
      all_files = (os.path.join(transl_dir, f) for f in os.listdir(transl_dir))
      start_time = min(
          os.path.getmtime(f) for f in all_files if os.path.isfile(f))
      values = []
      if FLAGS.bleu_variant in ("uncased", "both"):
        values.append(tf.Summary.Value(
            tag="BLEU_uncased" + FLAGS.tag_suffix, simple_value=0))
      if FLAGS.bleu_variant in ("cased", "both"):
        values.append(tf.Summary.Value(
            tag="BLEU_cased" + FLAGS.tag_suffix, simple_value=0))
      # 写入 BLEU=0 的事件
      writer.add_event(tf.summary.Event(summary=tf.Summary(value=values),
                                        wall_time=start_time, step=0))
      FLAGS.report_zero = False

    # 获取当前翻译文件的路径
    filename = transl_file.filename
    tf.logging.info("Evaluating " + filename)
    values = []
    
    # 计算不区分大小写的 BLEU 分数（如果启用）
    if FLAGS.bleu_variant in ("uncased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
                                          case_sensitive=False)
      values.append(tf.Summary.Value(tag="BLEU_uncased" + FLAGS.tag_suffix,
                                     simple_value=bleu))
      tf.logging.info("%s: BLEU_uncased = %6.2f" % (filename, bleu))
    
    # 计算区分大小写的 BLEU 分数（如果启用）
    if FLAGS.bleu_variant in ("cased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
                                          case_sensitive=True)
      values.append(tf.Summary.Value(tag="BLEU_cased" + FLAGS.tag_suffix,
                                     simple_value=bleu))
      tf.logging.info("%s: BLEU_cased = %6.2f" % (transl_file.filename, bleu))
    
    # 将评估结果写入 TensorBoard 事件文件
    writer.add_event(tf.summary.Event(
        summary=tf.Summary(value=values),
        wall_time=transl_file.mtime, step=transl_file.steps))
    # 刷新写入器确保数据立即写入磁盘
    writer.flush()
    
    # 更新 last_evaluated_step.txt 文件，记录当前评估的步数
    with open(last_step_file, "w") as ls_file:
      ls_file.write(str(transl_file.steps) + "\n")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
