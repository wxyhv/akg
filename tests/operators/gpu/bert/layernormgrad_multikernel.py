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
from comm_functions import test_single_out, test_multi_out

def compute_layernormgrad_1(data): 
    """
    data:[4096, 768]
    max_1: output1
    """
    max_1 = topi.sum(data, axis=(0), keepdims=False)

    return max_1

def compute_layernormgrad_2(data_1, data_2, data_3):
    """
    data_1:[4096, 768]
    data_2:[4096, 1]
    data_3:[4096, 1]
    red : output2
    """
    sub_1 = topi.subtract(data_1, data_2)
    add_1 = topi.add(data_3, 1e-11)
    rsqrt = topi.rsqrt(add_1)
    mul_1 = topi.multiply(1, rsqrt)
    mul_2 = topi.multiply(mul_1, sub_1)
    mul_3 = topi.multiply(mul_2, data_1)
    red = topi.sum(mul_3, axis=(0), keepdims=False)

    return red

def compute_layernormgrad_3(data_1, data_2, data_3, data_4, data_5):
    """
    data_1:[4096, 768]
    data_2:[4096, 1]
    data_3:[4096, 1]
    data_4:[768]
    data_5:[4096, 1]
    mul_11 mul_1２:[4096, 1]
    """
    sub_1 = topi.subtract(data_1, data_2)
    mul_1 = topi.multiply(-2, sub_1)
    red_1 = topi.sum(mul_1, axis=(-1), keepdims=True)

    add_1 = topi.add(data_3, 1e-11)
    sqrt_1 = topi.sqrt(add_1)
    mul_2 = topi.multiply(sqrt_1, add_1)
    div_1 = topi.divide(1, mul_2)
    mul_3 = topi.multiply(sub_1, div_1)
    mul_4 = topi.multiply(data_1, data_4)
    mul_5 = topi.multiply(mul_4, mul_3)
    mul_6 = topi.multiply(-0.5, mul_5)
    red_2 = topi.sum(mul_6, axis=(-1), keepdims=True)

    mul_7 = topi.multiply(red_2, red_1)
    mul_8 = topi.multiply(0.00130208, mul_7)
    red_3 = topi.sum(mul_4, axis=(-1), keepdims=True)

    mul_9 = topi.multiply(data_5, -1)
    mul_10 = topi.multiply(mul_9, red_3)
    add_2 = topi.add(mul_10, mul_8)
    mul_11 = topi.multiply(add_2, 0.00130208)
    mul_12 = topi.multiply(red_2, 0.00260417)
    
    return [mul_11, mul_12]
    
    
def compute_layernormgrad_4(mul_12, sub_1, mul_4, data_5, mul_11):
    """
    mul_12:[4096, 1]
    sub_1:[4096, 768]
    mul_4:[4096, 768]
    data_5:[4096, 1]
    mul_11:[4096, 1]
    add_4 output3: [4096, 768]
    """   
    mul_13 = topi.multiply(mul_12, sub_1)
    mul_14 = topi.multiply(mul_4, data_5)
    add_3 = topi.add(mul_14, mul_13)
    add_4 = topi.add(add_3, mul_11)

    return add_4

def compute_layernormgrad_expect_1(data): 
    """
    data:[4096, 768]
    max_1: output1
    """
    max_1 = np.sum(data, axis=(0), keepdims=False)

    return max_1

def compute_layernormgrad_expect_2(data_1, data_2, data_3):
    """
    data_1:[4096, 768]
    data_2:[4096, 1]
    data_3:[4096, 1]
    red : output2
    """
    sub_1 = np.subtract(data_1, data_2)
    add_1 = np.add(data_3, 1e-11)
    rsqrt = np.divide(1, np.sqrt(add_1))
    mul_1 = np.multiply(1, rsqrt)
    mul_2 = np.multiply(mul_1, sub_1)
    mul_3 = np.multiply(mul_2, data_1)
    red = np.sum(mul_3, axis=(0), keepdims=False)

    return red

def compute_layernormgrad_expect_3(data_1, data_2, data_3, data_4, data_5):
    """
    data_1:[4096, 768]
    data_2:[4096, 1]
    data_3:[4096, 1]
    data_4:[768]
    data_5:[4096, 1]
    mul_11 mul_1２:[4096, 1]
    """
    sub_1 = np.subtract(data_1, data_2)
    mul_1 = np.multiply(-2, sub_1)
    red_1 = np.sum(mul_1, axis=(-1), keepdims=True)

    add_1 = np.add(data_3, 1e-11)
    sqrt_1 = np.sqrt(add_1)
    mul_2 = np.multiply(sqrt_1, add_1)
    div_1 = np.divide(1, mul_2)
    mul_3 = np.multiply(sub_1, div_1)
    mul_4 = np.multiply(data_1, data_4)
    mul_5 = np.multiply(mul_4, mul_3)
    mul_6 = np.multiply(-0.5, mul_5)
    red_2 = np.sum(mul_6, axis=(-1), keepdims=True)

    mul_7 = np.multiply(red_2, red_1)
    mul_8 = np.multiply(0.00130208, mul_7)
    red_3 = np.sum(mul_4, axis=(-1), keepdims=True)

    mul_9 = np.multiply(data_5, -1)
    mul_10 = np.multiply(mul_9, red_3)
    add_2 = np.add(mul_10, mul_8)
    mul_11 = np.multiply(add_2, 0.00130208)
    mul_12 = np.multiply(red_2, 0.00260417)
    
    return [mul_11, mul_12]
    
    
def compute_layernormgrad_expect_4(mul_12, sub_1, mul_4, data_5, mul_11):
    """
    mul_12:[4096, 1]
    sub_1:[4096, 768]
    mul_4:[4096, 768]
    data_5:[4096, 1]
    mul_11:[4096, 1]
    add_4 output3: [4096, 768]
    """   
    mul_13 = np.multiply(mul_12, sub_1)
    mul_14 = np.multiply(mul_4, data_5)
    add_3 = np.add(mul_14, mul_13)
    add_4 = np.add(add_3, mul_11)

    return add_4

def test_1(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernormgrad_1, [shape], [dtype],
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernormgrad_expect_1(data)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]
    test_single_out(mod, [data], [output], expect)

def test_2(shape_1, shape_2, shape_3, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    data_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernormgrad_2, [shape_1, shape_2, shape_3], [dtype] * 3,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernormgrad_expect_2(data_1, data_2, data_3)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]
    test_single_out(mod, [data_1, data_2, data_3], [output], expect)

def test_3(shape_1, shape_2, shape_3, shape_4, shape_5, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    data_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    data_4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])
    data_5 = random_gaussian(shape_5, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernormgrad_3, [shape_1, shape_2, shape_3, shape_4, shape_5], [dtype] * 5,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernormgrad_expect_3(data_1, data_2, data_3, data_4, data_5)
    output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    test_multi_out(mod, [data_1, data_2, data_3, data_4, data_5], output, expect)

def test_4(shape_1, shape_2, shape_3, shape_4, shape_5, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    data_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    data_4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])
    data_5 = random_gaussian(shape_5, miu=1, sigma=0.1).astype(support_list[dtype])

    mod = utils.op_build(compute_layernormgrad_4, [shape_1, shape_2, shape_3, shape_4, shape_5], [dtype] * 5,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    expect = compute_layernormgrad_expect_4(data_1, data_2, data_3, data_4, data_5)
    output = [np.full(np.shape(expect), 0.0, expect.dtype)]
    test_single_out(mod, [data_1, data_2, data_3, data_4, data_5], output, expect)

if __name__ == "__main__":
    test_1([4096, 768], "float32")
    test_2([4096, 768], [4096, 1], [4096, 1], "float32")
    test_3([4096, 768], [4096, 1], [4096, 1], [768], [4096, 1], "float32")
    test_4([4096, 1], [4096, 768], [4096, 768], [4096, 1], [4096, 1], "float32")