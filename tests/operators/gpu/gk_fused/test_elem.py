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
from akg.ops.gk_fused_regis import elem_manual, elem_auto, elem_input_hrz_manual, elem_input_hrz_auto
from akg.ops.gk_fused_regis import elem_diamond_manual, elem_diamond_auto, elem_output_hrz_manual, elem_output_hrz_auto
from test_functions import test_single_out, test_multi_out

def gen_data(shape_lhs, shape_rhs, dtype, multi_out, func_name):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape_lhs, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape_rhs, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(lhs, rhs, multi_out, func_name)
    if isinstance(expect, (list, tuple)):
        output = [np.full(np.shape(e), np.nan, dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), np.nan, dtype)
    return lhs, rhs, output, expect

def elem(lhs, rhs, multi_out=False):
    expect = np.add(lhs, rhs)
    expect = np.multiply(expect, rhs)
    inter1 = np.add(lhs, lhs)
    inter1 = np.subtract(inter1, rhs)
    expect = np.divide(expect, inter1 + 1)
    if multi_out:
        expect1 = np.minimum(expect, lhs)
        expects = [expect, expect1]
    else:
        expects = expect
    return expects

def elem_input_hrz(lhs, rhs, multi_out=False):
    input1 = elem(lhs, lhs)
    input2 = elem(rhs, rhs)
    expect = elem(input1, input2, multi_out=multi_out)
    return expect

def elem_output_hrz(lhs, rhs, multi_out=False):
    inter1, inter2 = elem(lhs, rhs, multi_out=True)
    # if multi_out=False, length of expect is 2, else is 4
    output1 = elem(inter1, inter1, multi_out=multi_out)
    output2 = elem(inter2, inter2, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        expect = output1 + output2
    else:
        expect = [output1, output2]
    return expect

def elem_diamond(lhs, rhs, multi_out=False):
    inter1, inter2 = elem_output_hrz(lhs, rhs)
    expect = elem_input_hrz(inter1, inter2, multi_out=multi_out)
    return expect

def compute_expect(lhs, rhs, multi_out, func_name):
    expect = globals()[func_name](lhs, rhs, multi_out)
    return expect

def test_elem(shape_lhs, shape_rhs, dtype, multi_out=False, poly_sch=False, fusion_mode=''):
    shape_list = [shape_lhs, shape_rhs]
    dtype_list = [dtype, dtype]
    op_attrs = [multi_out]
    func_name = "elem_" + fusion_mode if fusion_mode != '' else "elem"
    if poly_sch:
        mod = utils.op_build(globals()[func_name + "_auto"], shape_list, dtype_list, op_attrs=op_attrs, attrs={"target":"cuda"})
    else:
        mod = utils.op_build(globals()[func_name + "_manual"], shape_list, dtype_list, op_attrs=op_attrs)

    lhs, rhs, output, expect = gen_data(shape_lhs, shape_rhs, dtype, multi_out, func_name)
    input = [lhs, rhs]
    if multi_out or fusion_mode == "output_hrz":
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)
