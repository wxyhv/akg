
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

"""fused operator dsl function: primitive + argmin"""
from __future__ import absolute_import
import akg.topi as topi
from .elem import elem

def prim_argmin(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    in_dtype = lhs.dtype
    if poly_sch==False and in_dtype == 'float16':
        lhs = topi.cast(lhs, 'float32')
        rhs = topi.cast(rhs, 'float32')
    output = elem(lhs, rhs)
    output_red = topi.argmin(output, axis=axis, keepdims=keepdims)
    if multi_out:
        output_red1 = topi.sum(output, axis=axis, keepdims=keepdims)
        if in_dtype=='float16' and poly_sch==False:
            output_red1 = topi.cast(output_red1, 'float16')
        outputs = [output_red, output_red1]
    else:
        outputs = output_red
    return outputs

def prim_argmin_input_hrz(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    input1 = elem(lhs, lhs)
    input2 = elem(rhs, rhs)
    output = prim_argmin(input1, input2, axis=axis, keepdims=keepdims, multi_out=multi_out)
    return output

def prim_argmin_output_hrz(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    output1 = prim_argmin(single_input, lhs, axis=axis, keepdims=keepdims, multi_out=multi_out)
    output2 = prim_argmin(single_input, rhs, axis=axis, keepdims=keepdims, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        output = output1 + output2
    else:
        output = [output1, output2]
    return output

def prim_argmin_diamond(lhs, rhs, axis=None, keepdims=False, multi_out=False, poly_sch=False):
    single_input = elem(lhs, rhs)
    inter1 = elem(single_input, lhs)
    inter2 = elem(single_input, rhs)
    output = prim_argmin(inter1, inter2, axis=axis, keepdims=keepdims, multi_out=multi_out)
    return output
