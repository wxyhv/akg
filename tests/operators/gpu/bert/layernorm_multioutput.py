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
from comm_functions import test_single_out

def compute_layernorm_1_out1(data1):
    """ 
    data1:(4096, 768)
    out: (4096, 1)
    """
    red_1 = topi.sum(data1, axis=(1), keepdims=True)
    return red_1

def compute_layernorm_1_out1_expect(data1):
    """ 
    data1:(4096, 768)
    out: (4096, 1)
    """
    red_1 = np.sum(data1, axis=(1), keepdims=True)
    return red_1

def compute_layernorm_2_out1(red_1, const_0=0.00130208):
    """ 
    red_1:(4096, 1)
    out: (4096, 1)
    """
    mul_1 = topi.multiply(red_1, const_0)
    return mul_1

def compute_layernorm_2_out1_expect(red_1, const_0=0.00130208):
    """ 
    red_1:(4096, 1)
    out: (4096, 1)
    """
    mul_1 = np.multiply(red_1, const_0)
    return mul_1

def compute_layernorm_3_out2(data1, mul_1):
    """ 
    data1: (4096, 768)
    mul_1: (4096, 1)
    out: (4096, 1)
    """
    sub_1 = topi.subtract(data1, topi.broadcast_to(mul_1, data1.shape))
    mul_2 = topi.multiply(sub_1, sub_1)
    red_2 = topi.sum(mul_2, axis=(1), keepdims=True)

    return red_2

def compute_layernorm_3_out2_expect(data1, mul_1):
    """ 
    data1: (4096, 768)
    mul_1: (4096, 1)
    out: (4096, 1)
    """
    sub_1 = np.subtract(data1, np.broadcast_to(mul_1, data1.shape))
    mul_2 = np.multiply(sub_1, sub_1)
    red_2 = np.sum(mul_2, axis=(1), keepdims=True)

    return red_2

def compute_layernorm_4_out2(red_2, const_0=0.00130208):
    """
    red_2:(4096, 1)
    out:(4096, 1)
    """
    mul_3 = topi.multiply(red_2, const_0)

    return mul_3

def compute_layernorm_4_out2_expect(red_2, const_0=0.00130208):
    """
    red_2:(4096, 1)
    out:(4096, 1)
    """
    mul_3 = np.multiply(red_2, const_0)

    return mul_3

def compute_layernorm_5_out3(data1, data2, data3, mul_3, mul_1, const_1=1e-07):
    """
    data1:(4096, 768)
    data2:(768,)
    data3:(768,)
    mul_3:(4096, 1)
    mul_1:(4096, 1)
    """
    add_1 = topi.add(mul_3, const_1)
    rsqrt_1 = topi.rsqrt(add_1)
    sub_1 = topi.subtract(data1, topi.broadcast_to(mul_1, data1.shape))
    mul_4 = topi.multiply(sub_1, rsqrt_1)
    brd_data2 = topi.expand_dims(data2, axis=(0))
    mul_5 = topi.multiply(brd_data2, mul_4)
    brd_data3 = topi.expand_dims(data3, axis=(0))
    add_2 = topi.add(mul_5, brd_data3)

    return add_2

def compute_layernorm_5_out3_expect(data1, data2, data3, mul_3, mul_1, const_1=1e-07):
    """
    data1:(4096, 768)
    data2:(768,)
    data3:(768,)
    mul_3:(4096, 1)
    mul_1:(4096, 1)
    """
    add_1 = np.add(mul_3, const_1)
    rsqrt_1 = np.divide(1, np.sqrt(add_1))
    sub_1 = np.subtract(data1, np.broadcast_to(mul_1, data1.shape))
    mul_4 = np.multiply(sub_1, rsqrt_1)
    brd_data2 = np.expand_dims(data2, axis=(0))
    mul_5 = np.multiply(brd_data2, mul_4)
    brd_data3 = np.expand_dims(data3, axis=(0))
    add_2 = np.add(mul_5, brd_data3)

    return add_2
"""
compute_layernorm_1_out1(data1)
compute_layernorm_2_out1(red_1, const_0=0.00130208)
compute_layernorm_3_out2(data1, mul_1)
compute_layernorm_4_out2(red_2, const_0=0.00130208)
compute_layernorm_5_out3(data1, data2, data3, mul_3, mul_1, const_1=1e-07)
"""
def test_compute_layernorm_1_out1(shape_1, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernorm_1_out1, [shape_1], [dtype],
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernorm_1_out1_expect(data1)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]

    test_single_out(mod, [data1], [output], expect)

def test_compute_layernorm_2_out1(shape_1, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernorm_2_out1, [shape_1], [dtype],
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernorm_2_out1_expect(data1)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]

    test_single_out(mod, [data1], [output], expect)

def test_compute_layernorm_3_out2(shape_1, shape_2, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernorm_3_out2, [shape_1, shape_2], [dtype, dtype],
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernorm_3_out2_expect(data1, data2)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]

    test_single_out(mod, [data1, data2], [output], expect)

def test_compute_layernorm_4_out2(shape_1, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernorm_4_out2, [shape_1], [dtype],
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernorm_4_out2_expect(data1)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]

    test_single_out(mod, [data1], [output], expect)

def test_compute_layernorm_5_out3(shape_1, shape_2, shape_3, shape_4, shape_5, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    data3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    data4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])
    data5 = random_gaussian(shape_5, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernorm_5_out3, [shape_1, shape_2, shape_3, shape_4, shape_5], [dtype] * 5,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernorm_5_out3_expect(data1, data2, data3, data4, data5)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]

    test_single_out(mod, [data1, data2, data3, data4, data5], [output], expect)

if __name__ == "__main__":
    test_compute_layernorm_1_out1((4096, 768), 'float32')
    test_compute_layernorm_2_out1((4096, 1), 'float32')
    test_compute_layernorm_3_out2((4096, 768), (4096, 1), 'float32')
    test_compute_layernorm_4_out2((4096, 1), 'float32')
    test_compute_layernorm_5_out3((4096, 768), (768,), (768,), (4096, 1), (4096, 1), 'float32')