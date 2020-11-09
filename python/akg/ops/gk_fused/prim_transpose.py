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

"fused operator dsl function: basic fusion + transpose "
from __future__ import absolute_import
import akg.topi as topi
from .elem import elem, elem_input_hrz, elem_output_hrz, elem_diamond

def prim_transpose(lhs, rhs, new_axis_order, multi_out, fusion_mode):

    # prim_res = elem(lhs, rhs)
    # output = topi.transpose(prim_res, axes=new_axis_order)
    # if multi_out:
    #     reverse_axis_order = new_axis_order
    #     reverse_axis_order.reverse()
    #     output1 = topi.transpose(prim_res, axes=reverse_axis_order)
    #     outputs = [output, output1]
    # else:
    #     outputs = output
    # return outputs    

    func_name = "elem_" + fusion_mode if fusion_mode != '' else "elem"
    elem_res = globals()[func_name](lhs, rhs, multi_out=multi_out)
    
    if not isinstance(elem_res, (list, tuple)):
        output = topi.transpose(elem_res, axes= new_axis_order)
    else:
        reverse_axis_order = new_axis_order.copy()
        reverse_axis_order.reverse()
        output = []
        for (i, tensor) in enumerate(elem_res):
            #if i & 1 == 0: #  output 0 and 2
            output.append(topi.transpose(tensor, axes=new_axis_order))
            #else: # output 1 and 3
            #    output.append(topi.transpose(tensor, axes=reverse_axis_order))

    return output