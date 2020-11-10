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
from akg.ops.gk_fused_regis import prim_unpad_manual, prim_unpad_auto, prim_unpad_input_hrz_manual, prim_unpad_input_hrz_auto
from akg.ops.gk_fused_regis import prim_unpad_diamond_manual, prim_unpad_diamond_auto, prim_unpad_output_hrz_manual, prim_unpad_output_hrz_auto

def gen_data(shape_lhs, shape_rhs, dtype, func_name, unpad_before, unpad_after, multi_out=False):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape_lhs, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape_rhs, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(lhs, rhs, func_name, unpad_before, unpad_after, multi_out)
    if isinstance(expect, (list, tuple)):
        output = [np.full(np.shape(e), np.nan, dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), np.nan, dtype)
    return lhs, rhs, output, expect

def prim_unpad(lhs, rhs, unpad_before, unpad_after, multi_out=False):
    expect_elem = compute_elem_expect(lhs, rhs)
    expect_red = np.max(expect_elem, axis=(0), keepdims=False)
    shape_red = np.shape(expect_red)
    expect = expect_elem[unpad_before[0]: shape_red[0] - unpad_after[0], unpad_before[1]: shape_red[1] - unpad_after[1], 
                            unpad_before[2]: shape_red[2] - unpad_after[2]]
    if multi_out:
        pad_width = list(zip(unpad_before, unpad_after))
        expect1 = np.pad(expect, pad_width, 'constant', constant_values=(1.0, 1.0))
        expects = [expect, expect1]
    else:
        expects = expect
    return expects

def prim_unpad_input_hrz(lhs, rhs, unpad_before, unpad_after, multi_out=False):
    input1 = compute_elem_expect(lhs, lhs)
    input2 = compute_elem_expect(rhs, rhs)
    expect = prim_unpad(input1, input2, unpad_before, unpad_after, multi_out=multi_out)
    return expect

def prim_unpad_output_hrz(lhs, rhs, unpad_before, unpad_after, multi_out=False):
    single_input = compute_elem_expect(lhs, rhs)
    output1 = prim_unpad(single_input, lhs, unpad_before, unpad_after, multi_out=multi_out)
    output2 = prim_unpad(single_input, rhs, unpad_before, unpad_after, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        expect = output1 + output2
    else:
        expect = [output1, output2]
    return expect

def prim_unpad_diamond(lhs, rhs, unpad_before, unpad_after, multi_out=False):
    single_input = compute_elem_expect(lhs, rhs)
    inter1 = compute_elem_expect(single_input, lhs)
    inter2 = compute_elem_expect(single_input, rhs)
    expect = prim_unpad(inter1, inter2, unpad_before, unpad_after, multi_out=multi_out)
    return expect

def compute_expect(lhs, rhs, func_name, unpad_before, unpad_after, multi_out=False):
    expect = globals()[func_name](lhs, rhs, unpad_before, unpad_after, multi_out)
    return expect

def test_prim_unpad(shape_lhs, shape_rhs, dtype, unpad_before, unpad_after, multi_out=False, poly_sch=False, fusion_mode=''):
    shape_list = [shape_lhs, shape_rhs]
    dtype_list = [dtype, dtype]
    op_attrs = [unpad_before, unpad_after, multi_out]
    func_name = "prim_unpad_" + fusion_mode if fusion_mode != '' else "prim_unpad"
    if poly_sch:
        mod = utils.op_build(globals()[func_name + "_auto"], shape_list, dtype_list, op_attrs=op_attrs, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(globals()[func_name + "_manual"], shape_list, dtype_list, op_attrs=op_attrs)
    
    lhs, rhs, output, expect = gen_data(shape_lhs, shape_rhs, dtype, func_name, unpad_before, unpad_after, multi_out)
    input = [lhs, rhs]
    if multi_out or fusion_mode == "output_hrz":
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)