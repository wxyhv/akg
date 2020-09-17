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
# limitations under the License

"""fused_pad"""
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.fused_gpu import fused_pad

@akg.schedule(schedule_injective)
def fused_pad_manual(data, pad_before, pad_after, layout='NHWC', pad_value=0.0):
    """Fused_pad."""
    return fused_pad.fused_pad(data, pad_before, pad_after, layout, pad_value)

def fused_pad_auto(data, pad_before, pad_after, layout='NHWC', pad_value=0.0):
    """Fused_pad."""
    return fused_pad.fused_pad(data, pad_before, pad_after, layout, pad_value)