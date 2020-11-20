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

def compute_softmax_1(data):
    """ softmax op. """
    max_1 = topi.max(data, axis=(-1), keepdims=True)

    return max_1
def compute_softmax_2(data_1, data_2):
    sub_1 = topi.subtract(data_1, data_2)
    exp_1 = topi.exp(sub_1)
    
    return exp_1

def compute_softmax_3(data):
    sum_1 = topi.sum(data, axis=(-1), keepdims=True)

    return sum_1

def compute_softmax_4(data_1, data_2):
    div_1 = topi.divide(data_1, data_2)

    return div_1

def compute_softmax_expect_1(data):
    """ softmax op. """
    max_1 = np.max(data, axis=(-1), keepdims=True)

    return max_1
def compute_softmax_expect_2(data_1, data_2):
    sub_1 = np.subtract(data_1, data_2)
    exp_1 = np.exp(sub_1)
    
    return exp_1

def compute_softmax_expect_3(data):
    sum_1 = np.sum(data, axis=(-1), keepdims=True)

    return sum_1

def compute_softmax_expect_4(data_1, data_2):
    div_1 = np.divide(data_1, data_2)

    return div_1

def test_compute_softmax(shape_1, shape_2, dtype, func):
    func_map = {"comput_1":compute_softmax_1, "comput_2":compute_softmax_2, "comput_3":compute_softmax_3,
        "comput_4":compute_softmax_4}
    func_expect_map = {"comput_1":compute_softmax_expect_1, "comput_2":compute_softmax_expect_2, 
        "comput_3":compute_softmax_expect_3, "comput_4":compute_softmax_expect_4}
    multiple_input = ["comput_2", "comput_4"]

    support_list = {"float16": np.float16, "float32": np.float32}
    data_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])

    if func in multiple_input:
        expect = func_expect_map[func](data_1, data_2)
        shape_list = [shape_1, shape_2]
        dtype_list = [dtype, dtype]
        mod = utils.op_build(func_map[func], shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
        inputs = [data_1, data_2]
    else:
        expect = func_expect_map[func](data_1)
        shape_list = [shape_1]
        dtype_list = [dtype]
        mod = utils.op_build(func_map[func], shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
        inputs = [data_1]
    output = np.full(np.shape(expect), 0.0, expect.dtype)  
    test_single_out(mod, inputs, output, expect)

if __name__ == "__main__":
    test_compute_softmax((32, 12, 128, 128), (32, 12, 128, 1), 'float32', "comput_1")
    test_compute_softmax((32, 12, 128, 128), (32, 12, 128, 1), 'float32', "comput_2")
    test_compute_softmax((32, 12, 128, 128), (32, 12, 128, 1), 'float32', "comput_3")
    test_compute_softmax((32, 12, 128, 128), (32, 12, 128, 1), 'float32', "comput_4")
