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

"""fused operator dsl function: elemwise"""
from __future__ import absolute_import
import akg.topi as topi

def elem(lhs, rhs, multi_out=False):
    output = topi.add(lhs, rhs)
    output = topi.multiply(output, rhs)
    inter1 = topi.add(lhs, lhs)
    inter1 = topi.subtract(inter1, rhs)
    output = topi.divide(output, inter1+1)
    if multi_out:
        output1 = topi.minimum(output, lhs)
        outputs = [output, output1]
    else:
        outputs = output
    return outputs

def elem_input_hrz(data1, data2, multi_out=False):
    input1 = elem(data1, data1)
    input2 = elem(data2, data2)
    output = elem(input1, input2, multi_out=multi_out)
    return output

def elem_output_hrz(data1, data2, multi_out=False):
    inter1, inter2 = elem(data1, data2, multi_out=True)
    # if multi_out=False, num of outputs is 2, else is 4
    output1 = elem(inter1, inter1, multi_out=multi_out)
    output2 = elem(inter2, inter2, multi_out=multi_out)
    if isinstance(output1, list) and isinstance(output2, list):
        outputs = output1 + output2
    else:
        outputs = [output1, output2]
    return outputs

def elem_diamond(data1, data2, multi_out=False):
    inter1, inter2 = elem_output_hrz(data1, data2)
    output = elem_input_hrz(inter1, inter2, multi_out=multi_out)
    return output