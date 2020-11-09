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
import random
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.ops.gk_fused_regis import prim_transpose_manual, prim_transpose_auto
from test_elem import elem, elem_input_hrz, elem_output_hrz, elem_diamond
from test_functions import test_single_out, test_multi_out

def gen_data(shape_lhs, shape_rhs, dtype, new_axis_order, multi_out, func_name):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape_lhs, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape_rhs, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(lhs, rhs, new_axis_order, multi_out, func_name)
    if isinstance(expect, (list, tuple)):
        output = [np.full(np.shape(e), np.nan, dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), np.nan, dtype)
    return lhs, rhs, output, expect

def compute_expect(lhs, rhs,  new_axis_order, multi_out, func_name):
    # prim_res = compute_elem_expect(lhs, rhs)
    # output = np.transpose(prim_res, axes=tuple(new_axis_order))
 
    # if multi_out:
    #     reverse_axis_order = new_axis_order
    #     reverse_axis_order.reverse()
    #     output1 = np.transpose(prim_res, axes=tuple(reverse_axis_order))
    #     outputs = [output, output1]
    # else:
    #     outputs = output
    # return outputs
    elem_res = globals()[func_name](lhs, rhs, multi_out=multi_out)
        
    if not isinstance(elem_res, (list, tuple)):
        output = np.transpose(elem_res, axes= new_axis_order)
    else:
        reverse_axis_order = new_axis_order.copy()
        reverse_axis_order.reverse()
        print("new axis order 2: ", reverse_axis_order)
        output = []
        for (i, tensor) in enumerate(elem_res):
            #if i & 1 == 0: #  output 0 and 2
            output.append(np.transpose(tensor, axes=new_axis_order))
            #else: # output 1 and 3
                #output.append(np.transpose(tensor, axes=reverse_axis_order))

    return output

def test_prim_transpose(shape_lhs, shape_rhs, dtype, new_axis_order=None, multi_out=False, poly_sch=False, fusion_mode=''):
    shape_list = [shape_lhs, shape_rhs]
    dtype_list = [dtype, dtype]
    if new_axis_order is None:
        new_axis_order = list(range(len(shape_lhs)))
        random.shuffle(new_axis_order)
    print("new axis order 1: ", new_axis_order)
    op_attrs = [new_axis_order, multi_out, fusion_mode]
    func_name  = "elem_" + fusion_mode if fusion_mode != '' else "elem"
    if poly_sch:
        mod = utils.op_build(prim_transpose_auto, shape_list, dtype_list, op_attrs=op_attrs, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(prim_transpose_manual, shape_list, dtype_list, op_attrs=op_attrs)
    
    lhs, rhs, output, expect = gen_data(shape_lhs, shape_rhs, dtype, new_axis_order, multi_out, func_name)
    input = [lhs, rhs]
    if multi_out or fusion_mode == "output_hrz":
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

