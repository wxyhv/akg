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
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_multi_out

def compute_layernorm(data1, data2, data3, const_0=0.00130208, const_1=1e-07):
    """ layernorm op. """
    red_1 = topi.sum(data1, axis=(1), keepdims=True)
    mul_1 = topi.multiply(red_1, const_0)
    sub_1 = topi.subtract(data1, topi.broadcast_to(mul_1, data1.shape))
    mul_2 = topi.multiply(sub_1, sub_1)
    red_2 = topi.sum(mul_2, axis=(1), keepdims=True)
    mul_3 = topi.multiply(red_2, const_0)
    add_1 = topi.add(mul_3, const_1)
    rsqrt_1 = topi.rsqrt(add_1)
    mul_4 = topi.multiply(sub_1, rsqrt_1)
    brd_data2 = topi.expand_dims(data2, axis=(0))
    mul_5 = topi.multiply(brd_data2, mul_4)
    brd_data3 = topi.expand_dims(data3, axis=(0))
    add_2 = topi.add(mul_5, brd_data3)

    return [add_2, mul_1, mul_3]

def compute_layernorm_expect(data1, data2, data3, const_0=0.00130208, const_1=1e-07):
    red_1 = np.sum(data1, axis=(1), keepdims=True)
    mul_1 = np.multiply(red_1, const_0)
    sub_1 = np.subtract(data1, np.broadcast_to(mul_1, data1.shape))
    mul_2 = np.multiply(sub_1, sub_1)
    red_2 = np.sum(mul_2, axis=(1), keepdims=True)
    mul_3 = np.multiply(red_2, const_0)
    add_1 = np.add(mul_3, const_1)
    rsqrt_1 = np.divide(1, np.sqrt(add_1))
    mul_4 = np.multiply(sub_1, rsqrt_1)
    brd_data2 = np.expand_dims(data2, axis=(0))
    mul_5 = np.multiply(brd_data2, mul_4)
    brd_data3 = np.expand_dims(data3, axis=(0))
    add_2 = np.add(mul_5, brd_data3)

    return [add_2, mul_1, mul_3]

def gen_data(shape_1, shape_2, shape_3, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    data3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_layernorm_expect(data1, data2, data3)
    if isinstance(expect, (list, tuple)): 
        output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [data1, data2, data3]
    return input, output, expect


def test_compute_layernorm(shape_1, shape_2, shape_3, dtype):
    shape_list = [shape_1, shape_2, shape_3]
    dtype_list = [dtype] * 3
    
    mod = utils.op_build(compute_layernorm, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    input, output, expect = gen_data(shape_1, shape_2, shape_3, dtype)
    test_multi_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_layernorm((4096, 768), (768,), (768,), 'float32')