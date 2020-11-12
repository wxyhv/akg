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
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.topi.cuda.reduce_opt import schedule_reduce

"""manual schedule"""
@akg.schedule(schedule_reduce)
def compute_1088_manual(data1, data2):
    return compute_topi(data1, data2)

"""auto schedule"""
def compute_1088_auto(data1, data2):
    return compute_topi(data1, data2)

def compute_topi(data1, data2):
    data2 = topi.expand_dims(data2, axis=(1), num_newaxis=1)
    data_subtract = topi.subtract(data1, data2)
    data_exp = topi.exp(data_subtract)
    data_red = topi.sum(data_exp, axis=(1), keepdims=False)
    output = topi.log(data_red)
    return output

def compute_expect(data1, data2):
    data2 = np.expand_dims(data2, axis=1)
    data_subtract = np.subtract(data1, data2)
    data_exp = np.exp(data_subtract)
    data_red = np.sum(data_exp, axis=(1), keepdims=False)
    expect = np.log(data_red)
    return expect

def gen_data(shape_1, shape_2, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    data2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_expect(data1, data2)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [data1, data2]
    return input, output, expect


def test_compute_1088(shape_1, shape_2, out_dtype, multi_out=False, poly_sch=False):
    shape_list = [shape_1, shape_2]
    dtype_list = [out_dtype, out_dtype]
    if poly_sch:
        mod = utils.op_build(compute_1088_auto, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
    else:    
        mod = utils.op_build(compute_1088_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_1, shape_2, out_dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1088((32, 2), (32, ), 'float32', poly_sch=False)
    test_compute_1088((32, 2), (32, ), 'float32', poly_sch=True)