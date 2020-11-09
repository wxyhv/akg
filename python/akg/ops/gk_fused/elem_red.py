
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

"""fused operator dsl function: elemwise + reduce"""
from __future__ import absolute_import
import akg.topi as topi
from .elem import elem

def elem_red(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    in_dtype = lhs.dtype
    if poly_sch==False and in_dtype == 'float16':
        lhs = topi.cast(lhs, 'float32')
        rhs = topi.cast(rhs, 'float32')
    output = elem(lhs, rhs)
    output_red = topi.sum(output, axis=axis, keepdims=keepdims)
    if multi_out:
        output_red1 = topi.min(output, axis=axis, keepdims=keepdims)
        outputs = [output_red, output_red1]
    else:
        outputs = output_red
    if poly_sch==True:
        return outputs
    if in_dtype == 'float16':
        if isinstance(outputs, (list, tuple)):        
            outputs = [topi.cast(o, 'float16') for o in outputs]
        else:
            outputs = topi.cast(outputs, 'float16')
    return outputs

def elem_red_input_hrz(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    input1 = elem(lhs, lhs)
    input2 = elem(rhs, rhs)
    output = elem_red(input1, input2, axis=axis, keepdims=keepdims, multi_out=multi_out)
    return output

def elem_red_output_hrz(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    output1 = elem_red(single_input, lhs, axis=axis, keepdims=keepdims, multi_out=multi_out)
    output2 = elem_red(single_input, rhs, axis=axis, keepdims=keepdims, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        output = output1 + output2
    else:
        output = [output1, output2]
    return output

def elem_red_diamond(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    inter1 = elem(single_input, lhs)
    inter2 = elem(single_input, rhs)
    output = elem_red(inter1, inter2, axis=axis, keepdims=keepdims, multi_out=multi_out)
    return output
