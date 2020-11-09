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
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array

def test_single_out(mod, input, output, expect):
    arg_list = input + [output]
    output = utils.mod_launch(mod, arg_list, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    
    input = to_tvm_nd_array(input)
    expect = to_tvm_nd_array(expect)
    gpu_profiling(mod, *input, expect, 400)

def test_multi_out(mod, input, output, expect):
    arg_list = input + output
    output = utils.mod_launch(mod, arg_list, outputs=tuple(range(-len(output), 0)), expect=expect)
    res=True
    for i in range(len(output)):
        res &= np.allclose(output[i], expect[i], rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    
    input = to_tvm_nd_array(input)
    expect = to_tvm_nd_array(expect)
    gpu_profiling(mod, *input, *expect, 400)