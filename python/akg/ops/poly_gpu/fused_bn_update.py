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

"""fused_bn_update"""
from __future__ import absolute_import
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.fused_gpu import fused_bn_update

@akg.schedule(schedule_injective)
def fused_bn_update_manual(input1, input2, input3, input4, dtype="float32",
    c1=(1 / (256 * 7 * 7)), c2=1.001e-05, c3=1.00007975, c4=0.100000024):
    """
    Fuse_bn_update_manual
    """
    return fused_bn_update.fused_bn_update(input1, input2, input3, input4, dtype, c1, c2, c3, c4)

def fused_bn_update_auto(input1, input2, input3, input4, dtype="float32",
    c1=(1 / (256 * 7 * 7)), c2=1.001e-05, c3=1.00007975, c4=0.100000024):
    """
    Fused_bn_update_auto
    """
    return fused_bn_update.fused_bn_update(input1, input2, input3, input4, dtype, c1, c2, c3, c4)