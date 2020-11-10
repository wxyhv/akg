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

"""fused operator dsl function: primitive + unpadding"""
from __future__ import absolute_import
import akg.topi as topi
import numpy as np
from .elem import elem

def prim_unpad(lhs, rhs, unpad_before, unpad_after, multi_out=False, poly_sch=False):
    in_dtype = lhs.dtype
    if poly_sch==False and in_dtype == 'float16':
        lhs = topi.cast(lhs, 'float32')
        rhs = topi.cast(rhs, 'float32')
    output_elem = elem(lhs, rhs)
    output_red = topi.max(output_elem, axis=(0), keepdims=False)
    shape_red = output_red.shape
    unpad_after = np.subtract(shape_red, unpad_after)
    unpad_after = list(unpad_after)
    unpad_after = [int(i) for i in unpad_after]
    unpad_before = [int(i) for i in unpad_before]
    output = topi.strided_slice(output_red, unpad_before, unpad_after)
    if multi_out:
        output1 = topi.nn.pad(output, unpad_before, unpad_after, 1.0)
        outputs = [output, output1]
    else:
        outputs = output
    if poly_sch==True:
        return outputs
    if in_dtype == 'float16':
        if isinstance(outputs, (list, tuple)):        
            outputs = [topi.cast(o, 'float16') for o in outputs]
    else:
            outputs = topi.cast(outputs, 'float16')
    return outputs

def prim_unpad_input_hrz(lhs, rhs, unpad_before, unpad_after, multi_out=False, poly_sch=False):
    input1 = elem(lhs, lhs)
    input2 = elem(rhs, rhs)
    output = prim_unpad(input1, input2, unpad_before, unpad_after, multi_out=multi_out)
    return output

def prim_unpad_output_hrz(lhs, rhs, unpad_before, unpad_after, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    output1 = prim_unpad(single_input, lhs, unpad_before, unpad_after, multi_out=multi_out)
    output2 = prim_unpad(single_input, rhs, unpad_before, unpad_after, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        output = output1 + output2
    else:
        output = [output1, output2]
    return output

def prim_unpad_diamond(lhs, rhs, unpad_before, unpad_after, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    inter1 = elem(single_input, lhs)
    inter2 = elem(single_input, rhs)
    output = prim_unpad(inter1, inter2, unpad_before, unpad_after, multi_out=multi_out)
    return output