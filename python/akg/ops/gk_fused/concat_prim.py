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

def concat_prim(lhs, rhs, concat_axis, multi_out, fusion_mode):
    concat_res1 = topi.concatenate((lhs, lhs), axis=concat_axis)
    concat_res2 = topi.concatenate((rhs, rhs), axis=concat_axis)
    func_name = "elem_" + fusion_mode if fusion_mode != '' else "elem"
    output = globals()[func_name](concat_res1, concat_res2, multi_out=multi_out)
    return output