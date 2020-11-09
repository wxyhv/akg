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
from akg.ops.gk_fused_regis import concat_prim_manual, concat_prim_auto
from test_elem import elem, elem_input_hrz, elem_output_hrz, elem_diamond
from test_functions import test_single_out, test_multi_out

def gen_data(shape_lhs, shape_rhs, dtype, concat_axis, multi_out, func_name):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape_lhs, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape_rhs, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(lhs, rhs, concat_axis, multi_out, func_name)
    if isinstance(expect, (list, tuple)):
        output = [np.full(np.shape(e), np.nan, dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), np.nan, dtype)
    return lhs, rhs, output, expect

def compute_expect(lhs, rhs, concat_axis, multi_out, func_name):
    concat_res1 = np.concatenate((lhs, lhs), axis=concat_axis)
    concat_res2 = np.concatenate((rhs, rhs), axis=concat_axis)
    expect = globals()[func_name](concat_res1, concat_res2, multi_out)
    return expect

def test_concat_prim(shape_lhs, shape_rhs, dtype, concat_axis=0, multi_out=False, poly_sch=False, fusion_mode=''):
    shape_list = [shape_lhs, shape_rhs]
    dtype_list = [dtype, dtype]
    op_attrs = [concat_axis, multi_out, fusion_mode]
    func_name  = "elem_" + fusion_mode if fusion_mode != '' else "elem"
    if poly_sch:
        mod = utils.op_build(concat_prim_auto, shape_list, dtype_list, op_attrs=op_attrs, attrs={"target":"cuda"})
    else:
        mod = utils.op_build(concat_prim_manual, shape_list, dtype_list, op_attrs=op_attrs)

    lhs, rhs, output, expect = gen_data(shape_lhs, shape_rhs, dtype, concat_axis, multi_out, func_name)
    input = [lhs, rhs]
    if multi_out or fusion_mode == "output_hrz":
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)
