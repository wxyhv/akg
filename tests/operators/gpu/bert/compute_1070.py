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

""" XLA fused operator No.1070"""
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
    const_0 = topi.full([32, 12, 128, 128], "float32", 0.1)

    brd_4431 = topi.expand_dims(param_1, axis=3)
    div_289 = topi.divide(param_0, brd_4431)
    cmp_92 = topi.greater_equal(param_3, const_0)
    cvt_86 = topi.cast(cmp_92, "float32")
    mul_4351 = topi.multiply(param_2, cvt_86)
    mul_3179 = topi.multiply(div_289, mul_4351)
    red_501 = topi.sum(mul_3179, axis=3)

    return red_501

def compute_expect(param_0, param_1, param_2, param_3):
    const_0 = np.full([32, 12, 128, 128], 0.1, "float32")

    brd_4431 = np.expand_dims(param_1, axis=3)
    div_289 = np.divide(param_0, brd_4431)
    cmp_92 = np.greater_equal(param_3, const_0)
    cvt_86 = cmp_92.astype("float32")
    mul_4351 = np.multiply(param_2, cvt_86)
    mul_3179 = np.multiply(div_289, mul_4351)
    red_501 = np.sum(mul_3179, axis=3)

    return red_501

def compute_1070_auto(param_0, param_1, param_2, param_3):
    return compute_topi(param_0, param_1, param_2, param_3)

@akg.schedule(schedule_reduce)
def compute_1070_manual(param_0, param_1, param_2, param_3):
    return compute_topi(param_0, param_1, param_2, param_3)

def gen_data(shape_0, shape_1, shape_2, shape_3, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    param_0 = random_gaussian(shape_0, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    param_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    param_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1, param_2, param_3)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1, param_2, param_3]
    return input, output, expect

def test_compute_1070(shape_0, shape_1, shape_2, shape_3, dtype, multi_out=False, poly_sch=False):
    shape_list = [shape_0, shape_1, shape_2, shape_3]
    dtype_list = [dtype] * 4
    if poly_sch:
        mod = utils.op_build(compute_1070_auto, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
    else:    
        mod = utils.op_build(compute_1070_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_0, shape_1, shape_2, shape_3, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1070((32, 12, 128, 128), (32, 12, 128), (32, 12, 128, 128), (32, 12, 128, 128), 
    'float32', poly_sch=False)
    test_compute_1070((32, 12, 128, 128), (32, 12, 128), (32, 12, 128, 128), (32, 12, 128, 128), 
    'float32', poly_sch=True)
