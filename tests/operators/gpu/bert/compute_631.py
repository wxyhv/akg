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

""" XLA fused operator No.631"""
from __future__ import absolute_import
import numpy as np
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_single_out, test_multi_out
from akg.topi.cuda.reduce_opt import schedule_reduce
from akg.topi.cuda.injective_single_kernel import schedule_injective

def compute_topi(param_0, param_1,  param_2,  param_3,    param_4,   param_5,    param_6,
    param_7, param_8, param_9):
    const_0 = topi.full([640,], "float32", 0.00130208)
    const_1 = topi.full([640, 768], "float32", 0.134145)
    const_2 = topi.full([640, 768], "float32", 0.797885)
    const_3 = topi.full([640, 768], "float32", 0.5)
    const_4 = topi.full([640, 1], "float32", 2)
    const_5 = topi.full([640, 1], "float32", 0.00130208)
    const_6 = topi.full([640,], "float32", -0.5)
    const_7 = topi.full([640, 768], "float32", 1)

    mul_4315 = topi.multiply(param_3, param_1)
    mul_4314 = topi.multiply(param_2, const_0)
    brd_4538 = topi.broadcast_to(topi.expand_dims(mul_4314, axis=(1)), [640, 768])
    mul_4313 = topi.multiply(brd_4538, topi.negative(param_1))
    brd_3115 = topi.broadcast_to(topi.expand_dims(param_0, axis=(1)), [640, 768])
    mul_2261 = topi.multiply(brd_3115, topi.add(mul_4315, mul_4313))
    red_216 = topi.sum(mul_2261, axis=(0))
    red_560 = topi.sum(param_1, axis=(0))
    mul_3245 = topi.multiply(param_6, param_6)
    mul_3244 = topi.multiply(const_1, mul_3245)
    mul_4043 = topi.multiply(brd_3115, topi.broadcast_to(topi.expand_dims(param_9, axis=(0)), [640, 768]))
    mul_3243 = topi.multiply(mul_4043, param_1)
    btc_406 = topi.expand_dims(param_0, axis=(1))
    mul_3242 = topi.multiply(btc_406, btc_406)
    mul_3241 = topi.multiply(mul_3242, btc_406)
    mul_3240 = topi.multiply(param_8, const_6)
    mul_3239 = topi.multiply(mul_3241, topi.expand_dims(mul_3240, axis=1))
    mul_3237 = topi.multiply(const_4, topi.multiply(const_5, mul_3239))
    brd_757 = topi.broadcast_to(mul_3237, [640, 768])
    sub_263 = topi.subtract(param_3, brd_4538)
    mul_3236 = topi.multiply(brd_757, sub_263)
    add_1497 = topi.add(mul_3243, mul_3236)
    mul_3235 = topi.multiply(const_0, param_7)
    brd_756 = topi.broadcast_to(topi.expand_dims(mul_3235, axis=(1)), [640, 768])
    add_1496 = topi.add(add_1497, brd_756)
    mul_3234 = topi.multiply(const_3, add_1496)
    mul_3233 = topi.multiply(mul_3234, param_6)
    mul_3232 = topi.multiply(param_5, param_5)
    sub_262 = topi.subtract(const_7, mul_3232)
    mul_3231 = topi.multiply(mul_3233, sub_262)
    mul_3230 = topi.multiply(mul_3231, const_2)
    mul_3229 = topi.multiply(mul_3244, mul_3230)
    add_1495 = topi.add(mul_3229, mul_3230)
    mul_3228 = topi.multiply(param_4, add_1496)
    add_1494 = topi.multiply(add_1495, mul_3228)
    red_635 = topi.sum(add_1494, axis=(0))

    return [red_216, red_560, red_635, add_1494]


def compute_expect(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9):
    const_0 = np.full([640,], 0.00130208, "float32")
    const_1 = np.full([640, 768], 0.134145, "float32")
    const_2 = np.full([640, 768], 0.797885, "float32")
    const_3 = np.full([640, 768], 0.5, "float32")
    const_4 = np.full([640, 1], 2, "float32")
    const_5 = np.full([640, 1], 0.00130208, "float32")
    const_6 = np.full([640,], -0.5, "float32")
    const_7 = np.full([640, 768], 1, "float32")

    mul_4315 = np.multiply(param_3, param_1)
    mul_4314 = np.multiply(param_2, const_0)
    brd_4538 = np.broadcast_to(np.expand_dims(mul_4314, axis=(1)), [640, 768])
    mul_4313 = np.multiply(brd_4538, np.negative(param_1))
    brd_3115 = np.broadcast_to(np.expand_dims(param_0, axis=(1)), [640, 768])
    mul_2261 = np.multiply(brd_3115, np.add(mul_4315, mul_4313))
    red_216 = np.sum(mul_2261, axis=(0))
    red_560 = np.sum(param_1, axis=(0))
    mul_3245 = np.multiply(param_6, param_6)
    mul_3244 = np.multiply(const_1, mul_3245)
    mul_4043 = np.multiply(brd_3115, np.broadcast_to(np.expand_dims(param_9, axis=(0)), [640, 768]))
    mul_3243 = np.multiply(mul_4043, param_1)
    btc_406 = np.expand_dims(param_0, axis=(1))
    mul_3242 = np.multiply(btc_406, btc_406)
    mul_3241 = np.multiply(mul_3242, btc_406)
    mul_3240 = np.multiply(param_8, const_6)
    mul_3239 = np.multiply(mul_3241, np.expand_dims(mul_3240, axis=1))
    mul_3237 = np.multiply(const_4, np.multiply(const_5, mul_3239))
    brd_757 = np.broadcast_to(mul_3237, [640, 768])
    sub_263 = np.subtract(param_3, brd_4538)
    mul_3236 = np.multiply(brd_757, sub_263)
    add_1497 = np.add(mul_3243, mul_3236)
    mul_3235 = np.multiply(const_0, param_7)
    brd_756 = np.broadcast_to(np.expand_dims(mul_3235, axis=(1)), [640, 768])
    add_1496 = np.add(add_1497, brd_756)
    mul_3234 = np.multiply(const_3, add_1496)
    mul_3233 = np.multiply(mul_3234, param_6)
    mul_3232 = np.multiply(param_5, param_5)
    sub_262 = np.subtract(const_7, mul_3232)
    mul_3231 = np.multiply(mul_3233, sub_262)
    mul_3230 = np.multiply(mul_3231, const_2)
    mul_3229 = np.multiply(mul_3244, mul_3230)
    add_1495 = np.add(mul_3229, mul_3230)
    mul_3228 = np.multiply(param_4, add_1496)
    add_1494 = np.multiply(add_1495, mul_3228)
    red_635 = np.sum(add_1494, axis=(0))

    return [red_216, red_560, red_635, add_1494]


def compute_631_auto(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9):
    return compute_topi(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9)

@akg.schedule(schedule_injective)
def compute_631_manual(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9):
    return compute_topi(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9)

def gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6,shape_7,
    shape_8, shape_9, dtype):
    support_list = {"float16": np.float16, "float32": np.float16, "int32": np.int32}
    param_0 = random_gaussian(shape_0, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    param_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    param_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    param_4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])
    param_5 = random_gaussian(shape_5, miu=1, sigma=0.1).astype(support_list[dtype])
    param_6 = random_gaussian(shape_6, miu=1, sigma=0.1).astype(support_list[dtype])
    param_7 = random_gaussian(shape_7, miu=1, sigma=0.1).astype(support_list[dtype])
    param_8 = random_gaussian(shape_8, miu=1, sigma=0.1).astype(support_list[dtype])
    param_9 = random_gaussian(shape_9, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1, param_2, param_3, param_4, param_5, param_6,
    param_7, param_8, param_9)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1, param_2, param_3, param_4, param_5, param_6,
        param_7, param_8, param_9]
    return input, output, expect

def test_compute_631(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6,shape_7,
    shape_8, shape_9, dtype, multi_out=True, poly_sch=False):
    shape_list = [shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6,shape_7,
        shape_8, shape_9]
    dtype_list = [dtype] * 10
    if poly_sch:
        mod = utils.op_build(compute_631_auto, shape_list, dtype_list,
            attrs={"target":"cuda", "enable_akg_reduce_lib":True})
    else:    
        mod = utils.op_build(compute_631_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6,shape_7,
    shape_8, shape_9, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_631((640,), (640, 768), (640,), (640, 768), (640, 768), (640, 768), (640, 768),
        (640,), (640,), (768,), 'float32', poly_sch=False)
    test_compute_631((640,), (640, 768), (640,), (640, 768), (640, 768), (640, 768), (640, 768),
        (640,), (640,), (768,), 'float32', poly_sch=True)
