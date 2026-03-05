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

"""杂项工具函数。

包含各种通用的辅助函数和工具。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import re

# 驼峰命名转蛇形命名的正则表达式
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def camelcase_to_snakecase(name):
  """将驼峰命名转换为蛇形命名。

  参数：
      name: 驼峰命名的字符串

  返回：
      蛇形命名的字符串
  """


def snakecase_to_camelcase(name):
  return "".join([w[0].upper() + w[1:] for w in name.split("_")])


def pprint_hparams(hparams):
  """Represents hparams using its dictionary and calls pprint.pformat on it."""
  return "\n{}".format(pprint.pformat(hparams.values(), width=1))
