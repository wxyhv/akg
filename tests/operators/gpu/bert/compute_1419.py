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

""" XLA fused operator No.1419"""
from __future__ import absolute_import
import numpy as np
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_single_out, test_multi_out
from akg.topi.cuda.reduce_opt import schedule_reduce
from akg.topi.cuda.injective_single_kernel import schedule_injective

def compute_topi(param_0, param_1, param_2, param_3):
    const_0 = topi.full_like(param_0, 0.134145)
    const_1 = topi.full_like(param_0, 0.5)
    const_2 = topi.full_like(param_0, 1)
    const_3 = topi.full_like(param_0, 0.797885)

    mul_3205 = topi.multiply(param_3, param_3)
    mul_3204 = topi.multiply(mul_3205, const_0)
    mul_3203 = topi.multiply(param_3, const_1)
    mul_3202 = topi.multiply(mul_3203, param_1)
    mul_3201 = topi.multiply(param_2, param_2)
    sub_256 = topi.subtract(const_2, mul_3201)
    mul_3200 = topi.multiply(mul_3202, sub_256)
    mul_3199 = topi.multiply(const_3, mul_3200)
    mul_3198 = topi.multiply(mul_3204, mul_3199)
    add_1489 = topi.add(mul_3198, mul_3199)
    mul_3197 = topi.multiply(param_0, param_1)
    add_1488 = topi.multiply(add_1489, mul_3197)
    red_632 = topi.sum(add_1488, axis=(1))

    return [red_632, add_1488]

def compute_expect(param_0, param_1, param_2, param_3):
    const_0 = np.full_like(param_0, 0.134145)
    const_1 = np.full_like(param_0, 0.5)
    const_2 = np.full_like(param_0, 1)
    const_3 = np.full_like(param_0, 0.797885)

    mul_3205 = np.multiply(param_3, param_3)
    mul_3204 = np.multiply(mul_3205, const_0)
    mul_3203 = np.multiply(param_3, const_1)
    mul_3202 = np.multiply(mul_3203, param_1)
    mul_3201 = np.multiply(param_2, param_2)
    sub_256 = np.subtract(const_2, mul_3201)
    mul_3200 = np.multiply(mul_3202, sub_256)
    mul_3199 = np.multiply(const_3, mul_3200)
    mul_3198 = np.multiply(mul_3204, mul_3199)
    add_1489 = np.add(mul_3198, mul_3199)
    mul_3197 = np.multiply(param_0, param_1)
    add_1488 = np.multiply(add_1489, mul_3197)
    red_632 = np.sum(add_1488, axis=(1))

    return [red_632, add_1488]


def compute__1419_auto(param_0, param_1, param_2, param_3):
    return compute_topi(param_0, param_1, param_2, param_3)

@akg.schedule(schedule_injective)
def compute_1419_manual(param_0, param_1, param_2, param_3):
    return compute_topi(param_0, param_1, param_2, param_3)

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    param_0 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    param_2 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    param_3 = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1, param_2, param_3)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1, param_2, param_3]
    return input, output, expect

def test_compute_1419(shape, dtype, multi_out=True, poly_sch=False):
    shape_list = [shape] * 4
    dtype_list = [dtype] * 4
    if poly_sch:
        mod = utils.op_build(compute_1419_auto, shape_list, dtype_list, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(compute_1419_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1419((4096, 3072), 'float32', poly_sch=False)
    test_compute_1419((4096, 3072), 'float32', poly_sch=True)
