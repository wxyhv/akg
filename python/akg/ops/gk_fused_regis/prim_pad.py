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

"""prim_pad"""
import akg
import akg.topi as topi
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.gk_fused.prim_pad import prim_pad, prim_pad_input_hrz, prim_pad_output_hrz, prim_pad_diamond

"""depth fusion"""
# @akg.schedule(topi.cuda.schedule_injective)
@akg.schedule(schedule_injective)
def prim_pad_manual(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=False)

def prim_pad_auto(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=True)

"""input-horizontal fusion"""
@akg.schedule(schedule_injective)
def prim_pad_input_hrz_manual(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_input_hrz(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=False)

def prim_pad_input_hrz_auto(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_input_hrz(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=True)

"""output-horizontal fusion """
@akg.schedule(schedule_injective)
def prim_pad_output_hrz_manual(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_output_hrz(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=False)

def prim_pad_output_hrz_auto(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_output_hrz(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=True)

"""diamond fusion"""
@akg.schedule(schedule_injective)
def prim_pad_diamond_manual(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_diamond(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=False)

def prim_pad_diamond_auto(data0, data1, pad_before, pad_after, pad_value=0.0, multi_out=False):
    return prim_pad_diamond(data0, data1, pad_before, pad_after, pad_value=pad_value, multi_out=multi_out, poly_sch=True)
