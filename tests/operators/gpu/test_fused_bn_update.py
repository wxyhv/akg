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

import numpy as np
from akg.ops.poly_gpu import fused_bn_update_manual, fused_bn_update_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tensorio import compare_tensor

def gen_data(shape, dtype):
    data1 = random_gaussian(shape, miu=3, sigma=0.1).astype(dtype)
    data2 = random_gaussian(shape, miu=3, sigma=0.1).astype(dtype)
    data3 = random_gaussian(shape, miu=3, sigma=0.1).astype(dtype)
    data4 = random_gaussian(shape, miu=3, sigma=0.1).astype(dtype)

    return [data1, data2, data3, data4]

def compute_expect(input, c1, c2, c3, c4):
    mul1 = np.multiply(input[1], c1)
    mul2 = np.multiply(input[0], c1)
    mul3 = np.multiply(mul2, mul2)
    sigma2 = np.subtract(mul1, mul3)
    sqrt1 = np.sqrt(np.add(sigma2, c2))
    out1 = np.divide(1, sqrt1)
    mul4 = np.multiply(sigma2, c3)
    out2 = np.multiply(c4, np.subtract(input[2], mul4))
    out3 = np.multiply(c4, np.subtract(input[3], mul2))

    return [out1, out2, out3]

def test_fused_bn_update(shape, dtype, c1=(1 / (256 * 7 * 7)), c2=1.001e-05, c3=1.00007975, c4=0.100000024, poly_sch=False):
    input = gen_data(shape, dtype)
    expect = compute_expect(input, c1, c2, c3, c4)
    attrs = [dtype, c1, c2, c3, c4]
    shapes = [input[0].shape] * 4
    dtypes = [dtype] * 4
    if poly_sch:
        mod = utils.op_build(fused_bn_update_auto, shapes, dtypes, op_attrs=attrs, attrs={"target": "cuda"})
    else:
        mod = utils.op_build(fused_bn_update_manual, shapes, dtypes, op_attrs=attrs)
    outputs =  [np.full(shape, np.nan, dtype)] * 3
    attrs_list =  input + outputs
    output = utils.mod_launch(mod, attrs_list, outputs=(range(-len(outputs), 0)))
    ret = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if ret else "Failed"))
    if not ret:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    data = to_tvm_nd_array(input)
    expect = to_tvm_nd_array(expect)
    gpu_profiling(mod, *data, *expect, 400)