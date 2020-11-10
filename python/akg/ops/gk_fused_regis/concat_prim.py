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

"""concat_prim"""
import akg
import akg.topi as topi
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.gk_fused.concat_prim import concat_prim

# """depth fusion"""
# @akg.schedule(topi.cuda.schedule_injective)
@akg.schedule(schedule_injective)
def concat_prim_manual(data0, data1, concat_axis=0, multi_out=False, fusion_mode=''):
    """Fused operater: concat_prim, with manual schedule"""
    return concat_prim(data0, data1, concat_axis, multi_out, fusion_mode)

def concat_prim_auto(data0, data1, concat_axis=0, multi_out=False, fusion_mode=''):
    """Fused operater: concat_prim, with auto schedule"""
    return concat_prim(data0, data1, concat_axis, multi_out, fusion_mode)