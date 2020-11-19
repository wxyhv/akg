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

def compute_softmax(data):
    """ softmax op. """
    max_1 = topi.max(data, axis=(-1), keepdims=True)
    sub_1 = topi.subtract(data, max_1)
    exp_1 = topi.exp(sub_1)
    sum_1 = topi.sum(exp_1, axis=(-1), keepdims=True)
    div_1 = topi.divide(exp_1, sum_1)

    return div_1

def compute_softmax_expect(data):
    max_1 = np.max(data, axis=(-1), keepdims=True)
    sub_1 = np.subtract(data, max_1)
    exp_1 = np.exp(sub_1)
    sum_1 = np.sum(exp_1, axis=(-1), keepdims=True)
    div_1 = np.divide(exp_1, sum_1)

    return div_1

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_softmax_expect(data)

    if isinstance(expect, (list, tuple)): 
        output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)

    return data, output, expect


def test_compute_softmax(shape, dtype):
    shape_list = [shape]
    dtype_list = [dtype]
    
    mod = utils.op_build(compute_softmax, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})

    input, output, expect = gen_data(shape, dtype)
    test_single_out(mod, [input], output, expect)

if __name__ == "__main__":
    test_compute_softmax((32, 12, 128, 128), 'float32')