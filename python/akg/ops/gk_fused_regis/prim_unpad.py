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

"""prim_unpad"""
import akg
import akg.topi as topi
from akg.ops.gk_fused.prim_unpad import prim_unpad, prim_unpad_input_hrz, prim_unpad_output_hrz, prim_unpad_diamond

"""depth fusion"""
@akg.schedule(topi.cuda.schedule_injective)
def prim_unpad_manual(data0, data1, unpad_before, unpad_after, multi_out=False):
    """Fused operater: prim_unpad, with manual schedule"""
    return prim_unpad(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=False)

def prim_unpad_auto(data0, data1, unpad_before, unpad_after, multi_out=False):
    """Fused operater: prim_unpad, with auto schedule"""
    return prim_unpad(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=True)

"""input-horizontal fusion"""
@akg.schedule(topi.cuda.schedule_injective)
def prim_unpad_input_hrz_manual(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_input_hrz(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=False)

def prim_unpad_input_hrz_auto(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_input_hrz(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=True)

"""output-horizontal fusion """
@akg.schedule(topi.cuda.schedule_injective)
def prim_unpad_output_hrz_manual(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_output_hrz(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=False)

def prim_unpad_output_hrz_auto(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_output_hrz(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=True)

"""diamond fusion"""
@akg.schedule(topi.cuda.schedule_injective)
def prim_unpad_diamond_manual(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_diamond(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=False)

def prim_unpad_diamond_auto(data0, data1, unpad_before, unpad_after, multi_out=False):
    return prim_unpad_diamond(data0, data1, unpad_before, unpad_after, multi_out=multi_out, poly_sch=True)
