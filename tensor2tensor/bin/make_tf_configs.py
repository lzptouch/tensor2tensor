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

"""输出命令行参数和 JSON 编码的 TF_CONFIGs。

用法：

`t2t-make-tf-configs --masters="server1:1234" --ps="server3:2134,server4:2334"`

每个作业输出一行到 stdout，首先是 masters，然后是参数服务器。
每行包含 TF_CONFIG，然后是制表符，然后是该作业的命令行标志。

如果只有一个 master，它将具有 `--sync` 标志。

功能说明：
- 用于分布式 TensorFlow 训练配置生成
- 支持同步（SYNC）和异步（ASYNC）两种分布式训练模式
- 生成每个 worker 和 parameter server 的配置信息
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("masters", "", "Comma-separated list of master addresses")
flags.DEFINE_string("ps", "", "Comma-separated list of ps addresses")


def main(_):
  """主函数：生成 TensorFlow 分布式训练配置。
  
  Args:
    _: 未使用的参数占位符
  
  流程说明：
  1. 解析 masters 和 ps 地址列表
  2. 判断是同步还是异步训练
  3. 构建集群配置
  4. 为每个节点生成 TF_CONFIG 和命令行参数
  """
  # 验证必需参数：必须提供 masters 和 ps
  if not (FLAGS.masters and FLAGS.ps):
    raise ValueError("Must provide --masters and --ps")

  # 将逗号分隔的地址字符串分割成列表
  masters = FLAGS.masters.split(",")
  ps = FLAGS.ps.split(",")

  # 判断是否为同步训练：只有一个 master 时为同步模式
  is_sync = len(masters) == 1
  if is_sync:
    # 同步模式：一个 master 作为 chief，多个 workers
    print("Assuming SYNC distributed training with a single master and %d "
          "workers" % len(ps))
    cluster = {"ps": ps, "master": masters}
  else:
    # 异步模式：一个 chief 和多个 workers
    print("Assuming ASYNC distributed training with %d workers and %d "
          "parameter servers" % (len(masters), len(ps)))
    cluster = {"ps": ps, "chief": [masters[0]], "worker": masters[1:]}

  # Trainer configs - 为每个 master 节点生成配置
  for idx, addr in enumerate(masters):
    # 构建命令行参数列表
    cmd_line_flags = [
        "--master=grpc://%s" % addr,  # 设置 master 的 gRPC 地址
        "--ps_replicas=%d" % len(ps),  # 参数服务器副本数
        "--worker_replicas=%d" % len(masters),  # worker 副本数
        "--worker_gpu=%d" % (0 if is_sync else 1),  # 同步模式下 GPU 设置为 0
        "--worker_id=%d" % idx,  # worker ID
        "--ps_gpu=%d" % (1 if is_sync else 0),  # 同步模式下 PS 使用 GPU
        "--sync" if is_sync else "",  # 同步模式标志
        "--schedule=train",  # 训练计划
    ]
    if is_sync:
      # 同步模式：任务类型为 master
      task_type = "master"
      cmd_line_flags.append("--worker_job='/job:master'")
    else:
      # 异步模式：第一个是 chief，其他是 worker
      if idx == 0:
        task_type = "chief"
        idx = 0
        cmd_line_flags.append("--worker_job='/job:chief'")
      else:
        task_type = "worker"
        idx -= 1
        cmd_line_flags.append("--worker_job='/job:worker'")

    # 构建 TF_CONFIG JSON 对象
    tf_config = json.dumps({
        "cluster": cluster,  # 集群配置
        "task": {
            "type": task_type,  # 任务类型（master/chief/worker）
            "index": idx  # 任务索引
        },
        "environment": "cloud",  # 环境类型
    })
    cmd_line_flags = " ".join(cmd_line_flags)
    # 输出：TF_CONFIG 和命令行参数（用制表符分隔）
    print("'%s'\t%s" % (tf_config, cmd_line_flags))

  # Std server configs - 为参数服务器生成配置
  for idx, addr in enumerate(ps):
    # 参数服务器的 TF_CONFIG
    tf_config = json.dumps({
        "cluster": cluster,  # 集群配置
        "task": {
            "type": "ps",  # 任务类型固定为 ps（parameter server）
            "index": idx  # PS 索引
        },
        "environment": "cloud",  # 环境类型
    })
    # 参数服务器的命令行参数：运行标准服务器
    cmd_line_flags = "--schedule=run_std_server"
    print("'%s'\t%s" % (tf_config, cmd_line_flags))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
