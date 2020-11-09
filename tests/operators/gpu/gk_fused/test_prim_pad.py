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
from __future__ import absolute_import
import numpy as np
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from test_functions import test_single_out, test_multi_out
from test_elem import elem as compute_elem_expect
from akg.ops.gk_fused_regis import prim_pad_manual, prim_pad_auto, prim_pad_input_hrz_manual, prim_pad_input_hrz_auto
from akg.ops.gk_fused_regis import prim_pad_diamond_manual, prim_pad_diamond_auto, prim_pad_output_hrz_manual, prim_pad_output_hrz_auto

def gen_data(shape_lhs, shape_rhs, dtype, func_name, pad_before, pad_after, pad_value=0.0, multi_out=False):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape_lhs, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape_rhs, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(lhs, rhs, func_name, pad_before, pad_after, pad_value, multi_out)
    if isinstance(expect, (list, tuple)):
        output = [np.full(np.shape(e), np.nan, dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), np.nan, dtype)
    return lhs, rhs, output, expect

def prim_pad(lhs, rhs, pad_before, pad_after, pad_value=0.0, multi_out=False):
    expect_elem = compute_elem_expect(lhs, rhs)
    expect_elem = np.max(expect_elem, axis=(0), keepdims=True)
    pad_width = list(zip(pad_before, pad_after))
    expect = np.pad(expect_elem, pad_width, 'constant', constant_values=(pad_value, pad_value))
    if multi_out:
        expect1 = np.pad(expect_elem, pad_width, 'constant', constant_values=(pad_value+1.0, pad_value+1.0))
        expects = [expect, expect1]
    else:
        expects = expect
    return expects

def prim_pad_input_hrz(lhs, rhs, pad_before, pad_after, pad_value=0.0, multi_out=False):
    input1 = compute_elem_expect(lhs, lhs)
    input2 = compute_elem_expect(rhs, rhs)
    expect = prim_pad(input1, input2, pad_before, pad_after, pad_value, multi_out=multi_out)
    return expect

def prim_pad_output_hrz(lhs, rhs, pad_before, pad_after, pad_value=0.0, multi_out=False):
    single_input = compute_elem_expect(lhs, rhs)
    output1 = prim_pad(single_input, lhs, pad_before, pad_after, pad_value, multi_out=multi_out)
    output2 = prim_pad(single_input, rhs, pad_before, pad_after, pad_value, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        expect = output1 + output2
    else:
        expect = [output1, output2]
    return expect

def prim_pad_diamond(lhs, rhs, pad_before, pad_after, pad_value=0.0, multi_out=False):
    single_input = compute_elem_expect(lhs, rhs)
    inter1 = compute_elem_expect(single_input, lhs)
    inter2 = compute_elem_expect(single_input, rhs)
    expect = prim_pad(inter1, inter2, pad_before, pad_after, pad_value, multi_out=multi_out)
    return expect

def compute_expect(lhs, rhs, func_name, pad_before, pad_after, pad_value=0.0, multi_out=False):
    expect = globals()[func_name](lhs, rhs, pad_before, pad_after, pad_value, multi_out)
    return expect

def test_prim_pad(shape_lhs, shape_rhs, dtype, pad_before, pad_after, pad_value=0.0, multi_out=False, poly_sch=False, fusion_mode=''):
    shape_list = [shape_lhs, shape_rhs]
    dtype_list = [dtype, dtype]
    op_attrs = [pad_before, pad_after, pad_value, multi_out]
    func_name = "prim_pad_" + fusion_mode if fusion_mode != '' else "prim_pad"
    if poly_sch:
        mod = utils.op_build(globals()[func_name + "_auto"], shape_list, dtype_list, op_attrs=op_attrs, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(globals()[func_name + "_manual"], shape_list, dtype_list, op_attrs=op_attrs)

    lhs, rhs, output, expect = gen_data(shape_lhs, shape_rhs, dtype, func_name, pad_before, pad_after, pad_value, multi_out)
    input = [lhs, rhs]
    if multi_out or fusion_mode == "output_hrz":
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)