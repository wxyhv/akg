# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pad"""
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
import akg.tvm as tvm
import akg.topi as topi

@akg.schedule(schedule_injective)
def pad_manual(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """pad with manual schedule."""
    return topi.nn.pad(data, pad_before, pad_after, pad_value, name)

def pad_auto(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """pad with auto schedule."""
    return topi.nn.pad(data, pad_before, pad_after, pad_value, name)