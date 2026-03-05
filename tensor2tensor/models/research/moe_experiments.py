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

"""MTF 中的语言建模实验。

实现混合专家（MoE）模型在机器翻译任务上的架构实验，
探索不同配置对模型性能的影响。
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import mtf_transformer
from tensor2tensor.models import mtf_transformer2
from tensor2tensor.models.research import moe
from tensor2tensor.utils import registry


@registry.register_hparams
def xmoe_tr_dense_2k():
  """在翻译任务上的一系列架构实验。

  在 8 核心设置上运行

  119M 参数，einsum=0.95e13

  返回：
      HParams 对象
  """
  hparams = mtf_transformer2.mtf_bitransformer_base()
  hparams.encoder_layers = ["self_att", "drd"] * 4
  hparams.decoder_layers = ["self_att", "enc_att", "drd"] * 4
  hparams.batch_size = 64
  hparams.shared_embedding_and_softmax_weights = True
  hparams.mesh_shape = "batch:8"
  return hparams


@registry.register_hparams
def xmoe_tr_dense_32k():
  """更大的 d_ff。

  623M 参数，einsum=3.42e13

  返回：
      HParams 对象
  """
  hparams = xmoe_tr_dense_2k()
  hparams.d_ff = 32768
  return hparams


@registry.register_hparams
def xmoe_tr_1d():
  """混合专家（16 个专家）。

  623M 参数，einsum=1.09e13

  返回：
      HParams 对象
  """
  hparams = xmoe_tr_dense_2k()
  hparams.encoder_layers = ["self_att", "moe_1d"] * 4
  hparams.decoder_layers = ["self_att", "enc_att", "moe_1d"] * 4
  hparams.layout = "batch:batch;experts:batch"
  hparams.moe_hidden_size = 2048
  hparams.moe_num_experts = 16
  return hparams


@registry.register_hparams
def xmoe_tr_2d():
  """二维混合专家（16 个专家）。

  623M 参数，einsum=1.09e13

  返回：
      HParams 对象
  """
  hparams = xmoe_tr_dense_2k()
  hparams.mesh_shape = "b0:2;b1:4"
  hparams.outer_batch_size = 4
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.encoder_layers = ["self_att", "moe_2d"] * 4
  hparams.decoder_layers = ["self_att", "enc_att", "moe_2d"] * 4
  hparams.moe_hidden_size = 2048
  hparams.moe_experts_x = 4
  hparams.moe_experts_y = 4
  return hparams


@registry.register_hparams
def xmoe_dense_4k():
  """语言模型的一系列架构实验。

  所有这些架构都在 languagemodel_lm1b8k_packed 上运行 32000 步。

  所有对数困惑度都是每token的 - 乘以 1.298 得到每词的

  结果：
  model             params(M)  einsum  alltoall  mxu-util  log-ppl
  xmoe_dense_4k     30         3.0e12  0         45%        3.31
  xmoe_dense_8k     46         4.7e12  0         49%        3.24
  xmoe_dense_64k    282        2.8e13  0                    3.06
  xmoe_top_2        282        4.0e12  3.4e8     36%        3.07
  xmoe_top_2_c15    282        4.5e12  4.0e8     38%        3.07
  xmoe_2d           282        5.3e12  7.6e8     34%        3.06

  以 4 倍批量大小训练：
  xmoe_2d_88        1090       2.1e13  3.0e9     24%        3.07

  注意：配置和代码可能会随时更改，恕不另行通知。

  返回：
      HParams 对象
  """
  hparams = mtf_transformer.mtf_transformer_base_lm()
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0

  # 以下超参数在所有这些实验中都是恒定的。
  hparams.batch_size = 128
  hparams.d_model = 512
  hparams.d_kv = 128
  hparams.num_heads = 4
  hparams.decoder_layers = ["att", "drd"] * 4
  hparams.shared_embedding_and_softmax_weights = False
  hparams.learning_rate_schedule = "rsqrt_decay"

  # 我们将改变以下与 ffn/moe 层相关的参数。
  hparams.d_ff = 4096
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:8"
  return hparams


@registry.register_hparams
def xmoe_dense_8k():
  """d_ff=8192 的密集模型。

  返回：
      HParams 对象
  """
  hparams = xmoe_dense_4k()
  hparams.d_ff = 8192
  return hparams


@registry.register_hparams
def xmoe_dense_64k():
  """非常宽的层 - 在 4x4 上运行。

  返回：
      HParams 对象
  """
  hparams = xmoe_dense_4k()
  hparams.d_ff = 65536
  hparams.mesh_shape = "model:4,batch:8"
  return hparams


@registry.register_hparams
def xmoe_top_2():
  """混合专家（16 个专家）。

  返回：
      HParams 对象
  """
  hparams = xmoe_dense_4k()
  moe.set_default_moe_hparams(hparams)
  hparams.mesh_shape = "all:8"
  hparams.layout = "batch:all;experts:all"
  return hparams


@registry.register_hparams
def xmoe_top_2_c15():
  """混合专家（容量因子为 1.5）。

  返回：
      HParams 对象
  """
  hparams = xmoe_top_2()
  hparams.moe_capacity_factor_train = 1.5
  return hparams


@registry.register_hparams
def xmoe_2d():
  """16 个专家的二维分层混合。

  返回：
      HParams 对象
  """
  hparams = xmoe_top_2()
  hparams.decoder_layers = ["att", "hmoe"] * 4
  hparams.mesh_shape = "b0:2;b1:4"
  hparams.outer_batch_size = 4
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.moe_num_experts = [4, 4]
  return hparams


@registry.register_hparams
def xmoe_2d_debug():
  """用于调试。

  在 TPU 上运行此模型时，如果没有将 alltoall 强制转换为 bfloat16 的技巧，
  会在第一步导致 nan。
  TODO(noam): 调试

  返回：
      HParams 对象
  """
  hparams = xmoe_2d()
  hparams.decoder_layers = ["hmoe"] * 1
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def xmoe_2d_c15():
  """混合专家（容量因子为 1.5）。

  返回：
      HParams 对象
  """
  hparams = xmoe_2d()
  hparams.moe_capacity_factor_train = 1.5
  return hparams


@registry.register_hparams
def xmoe_2d_x64():
  """64 个专家的二维分层混合。

  返回：
      HParams 对象
  """
  hparams = xmoe_2d()
  # hparams.mesh_shape = "b0:4;b1:8"
  hparams.outer_batch_size = 4
  hparams.moe_num_experts = [8, 8]
  return hparams


@registry.register_hparams
def xmoe2_dense(sz):
  """语言建模的一系列架构实验。

  比上面的模型更大。

  所有模型都在 1024 个token的序列上训练。

  我们假设无限的训练数据，因此不需要 dropout。
  我们在训练中处理 2^36 个token = 524288 步，批量大小为 128

  TODO(noam): 为这些实验找到足够大的数据集。

  你可以使用 languagemodel_wiki_noref_v32k_l1k，但这太小了，
  （1 个 epoch = ~46000 步）所以训练将覆盖大约 11 个 epoch。

  注意：配置和代码可能会随时更改，恕不另行通知。

  在 TPU 4x4 上运行 524288 步，除非另有说明。

  参数：
      sz: 整数

  返回：
      HParams 对象
  """
  hparams = mtf_transformer.mtf_transformer_paper_lm(sz)
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.max_length = 1024
  hparams.batch_size = 128
  hparams.learning_rate_schedule = "rsqrt_decay*linear_decay"
  hparams.learning_rate_decay_steps = 65536
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def xmoe2_dense_0():
  """规模为 0 的密集模型。

  返回：
      HParams 对象
  """
  return xmoe2_dense(0)


@registry.register_hparams
def xmoe2_dense_1():
  """规模为 1 的密集模型。

  返回：
      HParams 对象
  """
  return xmoe2_dense(1)


@registry.register_hparams
def xmoe2_dense_2():
  """规模为 2 的密集模型。

  返回：
      HParams 对象
  """
  return xmoe2_dense(2)


@registry.register_hparams
def xmoe2_dense_3():
  """规模为 3 的密集模型。

  返回：
      HParams 对象
  """
  return xmoe2_dense(3)


@registry.register_hparams
def xmoe2_v1():
  """包含混合专家和局部注意力的模型。

  ~6B 参数

  3 个分层 moe 层中有 32 个专家。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense(0)
  moe.set_default_moe_hparams(hparams)
  hparams.decoder_layers = (
      ["local_att", "local_att", "drd",
       "att", "drd", "local_att", "local_att", "hmoe"] * 4)[:-1]
  hparams.d_ff = 2048
  hparams.d_kv = 128
  hparams.moe_hidden_size = 32768
  hparams.mesh_shape = "b0:4;b1:8"
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.outer_batch_size = 4
  hparams.moe_num_experts = [8, 4]
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def xmoe2_v1_x128():
  """128 个专家，~25B 参数 - 在 8x8 上训练 131072 步。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1()
  hparams.moe_num_experts = [16, 8]
  hparams.outer_batch_size = 8
  hparams.mesh_shape = "b0:8;b1:16"
  hparams.batch_size = 512
  hparams.learning_rate_decay_steps = 16384
  return hparams


@registry.register_hparams
def xmoe2_tiny():
  """在本地 CPU 上测试。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1()
  hparams.decoder_layers = [
      "local_att", "att", "compressed_att", "drd", "hmoe"]
  hparams.d_model = 128
  hparams.moe_hidden_size = 512
  hparams.outer_batch_size = 0
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def xmoe2_v1_l4k():
  """序列长度为 4096。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1()
  hparams.batch_size = 32
  hparams.max_length = 4096
  hparams.split_to_length = 4096
  hparams.reshape_logits_hack = True
  return hparams


@registry.register_hparams
def xmoe2_v1_l4k_local_only():
  """序列长度为 4096，只使用局部注意力。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1_l4k()
  hparams.decoder_layers = [
      "local_att" if l == "att" else l for l in hparams.decoder_layers]
  return hparams


@registry.register_hparams
def xmoe2_v1_l4k_global_only():
  """序列长度为 4096，只使用全局注意力。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1_l4k()
  hparams.decoder_layers = [
      "att" if l == "local_att" else l for l in hparams.decoder_layers]
  return hparams


@registry.register_hparams
def xmoe2_v1_l4k_compressed_c4():
  """使用压缩注意力，压缩因子为 4。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1_l4k()
  hparams.decoder_layers = [
      "compressed_att" if l == "att" else l for l in hparams.decoder_layers]
  hparams.compression_factor = 4
  return hparams


@registry.register_hparams
def xmoe2_v1_l4k_compressed_c8():
  """使用压缩注意力，压缩因子为 8。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1_l4k_compressed_c4()
  hparams.compression_factor = 8
  return hparams


@registry.register_hparams
def wiki_2x2_base():
  """一系列架构实验 - 在 2x2 上的维基百科语言模型。

  1 个 epoch = ~180k 步，批量大小为 32 - 我们可能永远不会完成一个 epoch！

  返回：
      HParams 对象
  """
  hparams = mtf_transformer.mtf_transformer_base_lm()
  hparams.shared_embedding_and_softmax_weights = False
  # 没有 dropout - 数据集足够大，可以避免过拟合。
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.max_length = 1024
  # 每个核心 4 个序列
  hparams.batch_size = 32
  # 我们在这些实验中不使用线性衰减，因为我们不希望
  # 在训练计划结束时质量急剧跳跃。
  # 一旦找到正确的架构，你可以插入这个。
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.mesh_shape = "all:8"
  hparams.layout = "batch:all;experts:all"

  # 混合专家的参数
  moe.set_default_moe_hparams(hparams)
  hparams.moe_num_experts = 16
  hparams.moe_hidden_size = 8192

  hparams.decoder_layers = ["att", "drd"] * 6
  hparams.d_model = 1024
  hparams.d_ff = 2048
  hparams.d_kv = 128
  hparams.num_heads = 4

  return hparams


@registry.register_hparams
def wiki_2x2_v1():
  """维基百科语言模型 v1 版本。

  返回：
      HParams 对象
  """
  hparams = wiki_2x2_base()
  hparams.decoder_layers = (
      ["local_att", "local_att", "drd",
       "att", "drd", "local_att", "local_att", "moe"] * 4)[:-1]
  return hparams


@registry.register_hparams
def wiki_2x2_local():
  """只使用局部注意力的维基百科语言模型。

  返回：
      HParams 对象
  """
  hparams = wiki_2x2_base()
  hparams.decoder_layers = ["local_att", "drd"] * 6
  return hparams


@registry.register_hparams
def denoise_m15():
  """去噪实验。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense_0()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {"type": "mask", "prob": 0.15}
  return hparams


@registry.register_hparams
def denoise_m30():
  """训练期间更多的掩码。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense_0()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {"type": "mask", "prob": 0.3}
  return hparams


@registry.register_hparams
def denoise_dense_2_m30():
  """训练期间更多的掩码（密集模型 2）。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense_2()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {"type": "mask", "prob": 0.3}
  return hparams


@registry.register_hparams
def denoise_z15():
  """替换token而不是掩码。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense_0()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {"type": "random_zipfian", "prob": 0.15}
  hparams.noising_use_eval_during_train = 0.25
  return hparams


@registry.register_hparams
def denoise_t15():
  """使用 dropout 和小型 transformer 进行噪声处理。

  返回：
      HParams 对象
  """
  hparams = xmoe2_dense_0()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {
      "type": "transformer",
      "overrides": {
          "noising_spec_train": {"type": "mask", "prob": 0.15},
          "noising_use_eval_during_train": 0.0,
          "decoder_layers": ["att", "drd"] * 4,
          "num_heads": 4,
          "d_model": 512,
          "d_ff": 2048,
      }
  }
  return hparams


@registry.register_hparams
def denoise_v1_m15():
  """去噪实验（v1 版本）。

  返回：
      HParams 对象
  """
  hparams = xmoe2_v1()
  # 没有局部注意力
  # TODO(noam): local-attention 的非掩码版本
  hparams.decoder_layers = [
      "att" if l == "local_att" else l for l in hparams.decoder_layers]
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = {"type": "mask", "prob": 0.15}
  return hparams


@registry.register_hparams
def denoise_v1_m30():
  """训练期间更多的掩码（v1 版本）。

  返回：
      HParams 对象
  """
  hparams = denoise_v1_m15()
  hparams.noising_spec_train = {"type": "mask", "prob": 0.3}
  return hparams


@registry.register_hparams
def denoise_v1_m50():
  """训练期间更多的掩码（v1 版本，掩码概率 0.5）。

  返回：
      HParams 对象
  """
  hparams = denoise_v1_m15()
  hparams.noising_spec_train = {"type": "mask", "prob": 0.5}
  return hparams


@registry.register_hparams
def denoise_v1_z15():
  """替换token而不是掩码（v1 版本）。

  返回：
      HParams 对象
  """
  hparams = denoise_v1_m15()
  hparams.noising_spec_train = {"type": "random_zipfian", "prob": 0.15}
  return hparams


@registry.register_hparams
def denoise_v1_t15():
  """使用 dropout 和小型 transformer 进行噪声处理（v1 版本）。

  返回：
      HParams 对象
  """
  hparams = denoise_v1_m15()
  hparams.noising_spec_train = {
      "type": "transformer",
      "overrides": {
          "noising_spec_train": {"type": "mask", "prob": 0.15},
          "noising_use_eval_during_train": 0.0,
          "decoder_layers": ["att", "drd"] * 4,
          "num_heads": 4,
          "d_model": 512,
          "d_ff": 2048,
      }
  }
  return hparams
