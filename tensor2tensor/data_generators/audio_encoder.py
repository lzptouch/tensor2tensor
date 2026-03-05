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

"""音频数据编码器。

包含用于编码和解码音频数据的类。
"""

import os
from subprocess import call
import tempfile
import numpy as np
from scipy.io import wavfile


class AudioEncoder(object):
  """用于保存和加载波形的编码器类。

  提供音频波形的编码和解码功能。
  """

  def __init__(self, num_reserved_ids=0, sample_rate=16000):
    """初始化音频编码器。

    参数：
        num_reserved_ids: 保留 id 数量（必须为 0）
        sample_rate: 采样率（默认 16000Hz）
    """
    assert num_reserved_ids == 0
    self._sample_rate = sample_rate

  @property
  def num_reserved_ids(self):
    """返回保留 id 数量。"""
    return 0

  def encode(self, s):
    """将包含文件名的字符串转换为 float32 列表。

    参数：
        s: 波形文件的路径

    返回：
        samples: int16 列表
    """
    def convert_to_wav(in_path, out_path, extra_args=None):
      if not os.path.exists(out_path):
        # TODO(dliebling) On Linux, check if libsox-fmt-mp3 is installed.
        args = ["sox", "--rate", "16k", "--bits", "16", "--channel", "1"]
        if extra_args:
          args += extra_args
        call(args + [in_path, out_path])

    # Make sure that the data is a single channel, 16bit, 16kHz wave.
    # TODO(chorowski): the directory may not be writable, this should fallback
    # to a temp path, and provide instructions for installing sox.
    if s.endswith(".mp3"):
      out_filepath = s[:-4] + ".wav"
      convert_to_wav(s, out_filepath, ["--guard"])
      s = out_filepath
    elif not s.endswith(".wav"):
      out_filepath = s + ".wav"
      convert_to_wav(s, out_filepath)
      s = out_filepath
    rate, data = wavfile.read(s)
    assert rate == self._sample_rate
    assert len(data.shape) == 1
    if data.dtype not in [np.float32, np.float64]:
      data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.tolist()

  def decode(self, ids):
    """Transform a sequence of float32 into a waveform.

    Args:
      ids: list of integers to be converted.

    Returns:
      Path to the temporary file where the waveform was saved.

    Raises:
      ValueError: if the ids are not of the appropriate size.
    """
    _, tmp_file_path = tempfile.mkstemp()
    wavfile.write(tmp_file_path, self._sample_rate, np.asarray(ids))
    return tmp_file_path

  def decode_list(self, ids):
    """Transform a sequence of int ids into a wavform file.

    Args:
      ids: list of integers to be converted.

    Returns:
      Singleton list: path to the temporary file where the wavfile was saved.
    """
    return [self.decode(ids)]

  @property
  def vocab_size(self):
    return 256
