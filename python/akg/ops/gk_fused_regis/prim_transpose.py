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

"""prim_transpose"""
import akg
from akg.ops.gk_fused.prim_transpose import prim_transpose

@akg.schedule(akg.topi.cuda.schedule_injective)
def prim_transpose_manual(lhs, rhs, new_axis_order=None, multi_out=False, fusion_mode=''):
    "Fused operator : prim_transpore, with manual schedule"
    return prim_transpose(lhs, rhs, new_axis_order, multi_out, fusion_mode)

def prim_transpose_auto(lhs, rhs, new_axis_order=None, multi_out=False, fusion_mode=''):
    "Fused operator : prim_transpore, with auto schedule"
    return prim_transpose(lhs, rhs, new_axis_order, multi_out, fusion_mode)