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

""" XLA fused operator No.1486"""
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
    const_0 = topi.full([32, 128], "float32", -10000)
    const_1 = topi.full([32, 128], "float32", 1)

    cvt_78 = topi.cast(param_1, "float32")
    sub_331 = topi.subtract(const_1, cvt_78)
    mul_33527 = topi.multiply(const_0, sub_331)
    brd_852 = topi.expand_like(mul_33527, param_0, axis=(1, 2))
    add_1743 = topi.add(param_0, brd_852)
    red_687 = topi.max(add_1743, axis=(3))

    return [red_687, add_1743, brd_852]

def compute_expect(param_0, param_1):
    const_0 = np.full([32, 128], -10000, "float32")
    const_1 = np.full([32, 128], 1, "float32")

    cvt_78 = param_1.astype("float32")
    sub_331 = np.subtract(const_1, cvt_78)
    mul_33527 = np.multiply(const_0, sub_331)
    brd_852 = np.expand_dims(mul_33527, axis=(1, 2))
    add_1743 = np.add(param_0, brd_852)
    red_687 = np.max(add_1743, axis=(3))

    return [red_687, add_1743, brd_852]

def compute_1486_auto(param_0, param_1):
    return compute_topi(param_0, param_1)

@akg.schedule(schedule_injective)
def compute_1486_manual(param_0, param_1):
    return compute_topi(param_0, param_1)

def gen_data(shape_0, shape_1, dtype_0, dtype_1):
    support_list = {"float16": np.float16, "float32": np.float16, "int32": np.int32}
    param_0 = random_gaussian(shape_0, miu=1, sigma=0.1).astype(support_list[dtype_0])
    param_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype_1])

    expect = compute_expect(param_0, param_1)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1]
    return input, output, expect

def test_compute_1486(shape_0, shape_1, dtype_0, dtype_1, multi_out=True, poly_sch=False):
    shape_list = [shape_0, shape_1]
    dtype_list = [dtype_0, dtype_1]
    if poly_sch:
        mod = utils.op_build(compute_1486_auto, shape_list, dtype_list, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(compute_1486_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_0, shape_1, dtype_0, dtype_1)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1486((32, 12, 128, 128), (32, 128), 'float32', 'int32', poly_sch=False)
    test_compute_1486((32, 12, 128, 128), (32, 128), 'float32', 'int32', poly_sch=True)
