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

""" XLA fused operator No.1425"""
from __future__ import absolute_import
import numpy as np
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_single_out, test_multi_out
from akg.topi.cuda.reduce_opt import schedule_reduce
from akg.topi.cuda.injective_single_kernel import schedule_injective

def compute_topi(param_0, param_1, param_2, param_3, param_4):
    brd_4507 = topi.expand_dims(param_4, axis=1)
    sub_343 = topi.subtract(param_3, topi.broadcast_to(brd_4507, [640, 21128]))
    brd_4506 = topi.expand_dims(param_2, axis=1)
    sub_342 = topi.subtract(sub_343, topi.broadcast_to(brd_4506, [640, 21128]))
    exp_2 = topi.exp(sub_342)
    mul_3251 = topi.multiply(exp_2, topi.broadcast_to(topi.expand_dims(param_1, axis=1), [640, 21128]))
    sub_264 = topi.subtract(param_0, mul_3251)
    red_638 = topi.sum(sub_264, axis=(0))

    return [red_638, sub_264]

def compute_expect(param_0, param_1, param_2, param_3, param_4):
    brd_4507 = np.expand_dims(param_4, axis=1)
    sub_343 = np.subtract(param_3, np.broadcast_to(brd_4507, [640, 21128]))
    brd_4506 = np.expand_dims(param_2, axis=1)
    sub_342 = np.subtract(sub_343, np.broadcast_to(brd_4506, [640, 21128]))
    exp_2 = np.exp(sub_342)
    mul_3251 = np.multiply(exp_2, np.broadcast_to(np.expand_dims(param_1, axis=1), [640, 21128]))
    sub_264 = np.subtract(param_0, mul_3251)
    red_638 = np.sum(sub_264, axis=(0))

    return [red_638, sub_264]

def compute_1425_auto(param_0, param_1, param_2, param_3, param_4):
    return compute_topi(param_0, param_1, param_2, param_3, param_4)

@akg.schedule(schedule_injective)
def compute_1425_manual(param_0, param_1, param_2, param_3, param_4):
    return compute_topi(param_0, param_1, param_2, param_3, param_4)

def gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    param_0 = random_gaussian(shape_0, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    param_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    param_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    param_4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1, param_2, param_3, param_4)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1, param_2, param_3, param_4]
    return input, output, expect

def test_compute_1425(shape_0, shape_1, shape_2, shape_3, shape_4, dtype, multi_out=True, poly_sch=False):
    shape_list = [shape_0, shape_1, shape_2, shape_3, shape_4]
    dtype_list = [dtype] * 5
    if poly_sch:
        mod = utils.op_build(compute_1425_auto, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
    else:    
        mod = utils.op_build(compute_1425_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_1425((640, 21128), (640,), (640,), (640, 21128), (640,), 'float32', poly_sch=False)
    test_compute_1425((640, 21128), (640,), (640,), (640, 21128), (640,), 'float32', poly_sch=True)
