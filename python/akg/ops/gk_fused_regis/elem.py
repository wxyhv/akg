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

"""elem"""
import akg
import akg.topi as topi
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.gk_fused.elem import elem, elem_input_hrz, elem_output_hrz, elem_diamond

"""depth fusion"""
# @akg.schedule(topi.cuda.schedule_injective)
@akg.schedule(schedule_injective)
def elem_manual(data0, data1, multi_out=False):
    """Fused operater: elem, with manual schedule"""
    return elem(data0, data1, multi_out)

def elem_auto(data0, data1, multi_out=False):
    """Fused operater: elem, with auto schedule"""
    return elem(data0, data1, multi_out)

"""input-horizontal fusion"""
@akg.schedule(schedule_injective)
def elem_input_hrz_manual(data0, data1, multi_out=False):
    return elem_input_hrz(data0, data1, multi_out)

def elem_input_hrz_auto(data0, data1, multi_out=False):
    return elem_input_hrz(data0, data1, multi_out)

"""output-horizontal fusion """
@akg.schedule(schedule_injective)
def elem_output_hrz_manual(data0, data1, multi_out=False):
    return elem_output_hrz(data0, data1, multi_out)

def elem_output_hrz_auto(data0, data1, multi_out=False):
    return elem_output_hrz(data0, data1, multi_out)

"""diamond fusion"""
@akg.schedule(schedule_injective)
def elem_diamond_manual(data0, data1, multi_out=False):
    return elem_diamond(data0, data1, multi_out)

def elem_diamond_auto(data0, data1, multi_out=False):
    return elem_diamond(data0, data1, multi_out)
