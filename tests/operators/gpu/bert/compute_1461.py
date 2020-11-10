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

""" XLA fused operator No.1461"""
from __future__ import absolute_import
import numpy as np
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_single_out, test_multi_out
from akg.topi.cuda.reduce_opt import schedule_reduce
from akg.topi.cuda.injective_single_kernel import schedule_injective

def compute_topi(param_0, param_1):
    add_1738 = topi.add(param_0, param_1)
    red_667 = topi.max(add_1738, axis=(3))

    return [red_667, add_1738]

def compute_expect(param_0, param_1):
    add_1738 = np.add(param_0, param_1)
    red_667 = np.max(add_1738, axis=(3))

    return [red_667, add_1738]

def compute_1461_auto(param_0, param_1):
    return compute_topi(param_0, param_1)

@akg.schedule(schedule_injective)
def compute_1461_manual(param_0, param_1):
    return compute_topi(param_0, param_1)

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float16, "int32": np.int32}
    param_0 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1]
    return input, output, expect

def test_compute_1461(shape, dtype, multi_out=True, poly_sch=False):
    shape_list = [shape] * 2
    dtype_list = [dtype] * 2
    if poly_sch:
        mod = utils.op_build(compute_1461_auto, shape_list, dtype_list, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(compute_1461_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1461((32, 12, 128, 128), 'float32', poly_sch=False)
    test_compute_1461((32, 12, 128, 128), 'float32', poly_sch=True)
