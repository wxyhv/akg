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

""" XLA fused operator No.791"""
from __future__ import absolute_import
import numpy as np
import akg
import akg.topi as topi
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from comm_functions import test_single_out, test_multi_out
from akg.topi.cuda.reduce_opt import schedule_reduce
from akg.topi.cuda.injective_single_kernel import schedule_injective

def compute_topi(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7):
    const_0 = topi.full([4096,], "float32", 0.00130208)
    const_1 = topi.full([4096, 768], "float32", 1.11111)
    const_2 = topi.full([4096, 1], "float32", 2)
    const_3 = topi.full([4096, 1], "float32", 0.00130208)
    const_4 = topi.full([4096,], "float32", -0.5)
    const_5 = topi.full([4096, 768], "float32", 0.1)

    brd_3090 = topi.broadcast_to(topi.expand_dims(param_0, axis=1), [4096, 768])
    mul_4596 = topi.multiply(param_3, param_1)
    mul_4595 = topi.multiply(param_2, const_0)
    brd_4760 = topi.broadcast_to(topi.expand_dims(mul_4595, axis=1), [4096, 768])
    mul_4594 = topi.multiply(brd_4760, topi.negative(param_1))
    add_1678 = topi.add(mul_4596, mul_4594)
    mul_2486 = topi.multiply(brd_3090, add_1678)
    red_374 = topi.sum(mul_2486, axis=(0))
    red_547 = topi.sum(param_1, axis=(0))
    brd_3233 = topi.broadcast_to(topi.expand_dims(param_7, axis=0), [4096, 768])
    mul_3998 = topi.multiply(brd_3090, brd_3233)
    mul_2813 = topi.multiply(mul_3998, param_1)
    btc_291 = topi.expand_dims(param_0, axis=1)
    mul_2811 = topi.multiply(btc_291, btc_291)
    mul_2810 = topi.multiply(mul_2811, btc_291)
    mul_2808 = topi.multiply(param_6, const_4)
    btc_290 = topi.expand_dims(mul_2808, axis=1)
    mul_2807 = topi.multiply(mul_2810, btc_290)
    mul_2805 = topi.multiply(const_3, mul_2807)
    mul_2804 = topi.multiply(const_2, mul_2805)
    brd_704 = topi.broadcast_to(mul_2804, [4096, 768])
    sub_223 = topi.subtract(param_3, brd_4760)
    mul_2802 = topi.multiply(brd_704, sub_223)
    add_1422 = topi.multiply(mul_2813, mul_2802)
    mul_2801 = topi.multiply(const_0, param_5)
    add_1421 = topi.add(add_1422, topi.expand_dims(mul_2801, axis=1))
    mul_2799 = topi.multiply(const_1, add_1421)
    cmp_15 = topi.greater_equal(param_4, const_5)
    cvt_15 = topi.cast(cmp_15, "float32")
    mul_2798 = topi.multiply(mul_2799, cvt_15)
    red_583 = topi.sum(mul_2798, axis=0)

    return [red_374, red_547, red_583, mul_2798, add_1421]

def compute_expect(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7):
    const_0 = np.full([4096,], 0.00130208, "float32")
    const_1 = np.full([4096, 768], 1.11111, "float32")
    const_2 = np.full([4096, 1], 2, "float32")
    const_3 = np.full([4096, 1], 0.00130208, "float32")
    const_4 = np.full([4096,], -0.5, "float32")
    const_5 = np.full([4096, 768], 0.1, "float32")

    brd_3090 = np.broadcast_to(np.expand_dims(param_0, axis=1), [4096, 768])
    mul_4596 = np.multiply(param_3, param_1)
    mul_4595 = np.multiply(param_2, const_0)
    brd_4760 = np.broadcast_to(np.expand_dims(mul_4595, axis=1), [4096, 768])
    mul_4594 = np.multiply(brd_4760, np.negative(param_1))
    add_1678 = np.add(mul_4596, mul_4594)
    mul_2486 = np.multiply(brd_3090, add_1678)
    red_374 = np.sum(mul_2486, axis=(0))
    red_547 = np.sum(param_1, axis=(0))
    brd_3233 = np.broadcast_to(np.expand_dims(param_7, axis=0), [4096, 768])
    mul_3998 = np.multiply(brd_3090, brd_3233)
    mul_2813 = np.multiply(mul_3998, param_1)
    btc_291 = np.expand_dims(param_0, axis=1)
    mul_2811 = np.multiply(btc_291, btc_291)
    mul_2810 = np.multiply(mul_2811, btc_291)
    mul_2808 = np.multiply(param_6, const_4)
    btc_290 = np.expand_dims(mul_2808, axis=1)
    mul_2807 = np.multiply(mul_2810, btc_290)
    mul_2805 = np.multiply(const_3, mul_2807)
    mul_2804 = np.multiply(const_2, mul_2805)
    brd_704 = np.broadcast_to(mul_2804, [4096, 768])
    sub_223 = np.subtract(param_3, brd_4760)
    mul_2802 = np.multiply(brd_704, sub_223)
    add_1422 = np.multiply(mul_2813, mul_2802)
    mul_2801 = np.multiply(const_0, param_5)
    add_1421 = np.add(add_1422, np.expand_dims(mul_2801, axis=1))
    mul_2799 = np.multiply(const_1, add_1421)
    cmp_15 = np.greater_equal(param_4, const_5)
    cvt_15 = cmp_15.astype("float32")
    mul_2798 = np.multiply(mul_2799, cvt_15)
    red_583 = np.sum(mul_2798, axis=0)

    return [red_374, red_547, red_583, mul_2798, add_1421]

def compute_791_auto(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7):
    return compute_topi(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7)

@akg.schedule(schedule_injective)
def compute_791_manual(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7):
    return compute_topi(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7)

def gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    param_0 = random_gaussian(shape_0, miu=1, sigma=0.1).astype(support_list[dtype])
    param_1 = random_gaussian(shape_1, miu=1, sigma=0.1).astype(support_list[dtype])
    param_2 = random_gaussian(shape_2, miu=1, sigma=0.1).astype(support_list[dtype])
    param_3 = random_gaussian(shape_3, miu=1, sigma=0.1).astype(support_list[dtype])
    param_4 = random_gaussian(shape_4, miu=1, sigma=0.1).astype(support_list[dtype])
    param_5 = random_gaussian(shape_5, miu=1, sigma=0.1).astype(support_list[dtype])
    param_6 = random_gaussian(shape_6, miu=1, sigma=0.1).astype(support_list[dtype])
    param_7 = random_gaussian(shape_7, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = compute_expect(param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7)
    if isinstance(expect, (list, tuple)): 
            output = [np.full(np.shape(e), 0.0, e.dtype) for e in expect]
    else:
        output = np.full(np.shape(expect), 0.0, expect.dtype)
    input = [param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7]
    return input, output, expect

def test_compute_791(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7,
    dtype, multi_out=True, poly_sch=False):
    shape_list = [shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7]
    dtype_list = [dtype] * 8
    if poly_sch:
        mod = utils.op_build(compute_791_auto, shape_list, dtype_list, attrs={"target":"cuda"})
    else:    
        mod = utils.op_build(compute_791_manual, shape_list, dtype_list)
    
    input, output, expect = gen_data(shape_0, shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, 
    shape_7, dtype)
    if multi_out:
        test_multi_out(mod, input, output, expect)
    else:
        test_single_out(mod, input, output, expect)

if __name__ == "__main__":
    test_compute_791((4096,), (4096, 768), (4096,), (4096, 768), (4096, 768), (4096,), (4096,),
    (768,), 'float32', poly_sch=False)
    test_compute_791((4096,), (4096, 768), (4096,), (4096, 768), (4096, 768), (4096,), (4096,),
    (768,), 'float32', poly_sch=True)
